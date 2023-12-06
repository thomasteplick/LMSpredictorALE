/*
    Create an LMS Predictor for Adaptive Line Enhancement.  Choose sinusoids,
    AM, or FM signals to predict.  Plot the impulse and frequency responses
	of the filter.  Filter noisy signal using the predictor.

	Plot the LMS adaptive filter in complex(real, imag) from a file, one complex
	number  per line.
	r1 i1
	r2 i2
	...
	rn in

	The html/template package is used to generate the html sent to the client.
	Use CSS display grid to display a 300x300 grid of cells.
	Use CSS flexbox to display the labels on the x and y axes.

	Calculate the power spectral density (PSD) using Welch periodogram spectral
	estimation and plot it.  Average the periodograms using 50% overlap with a
	user-chosen window function such as Bartlett, Hanning, Hamming, or Welch.
*/

package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"math/rand"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"text/template"

	"github.com/mjibson/go-dsp/fft"
)

const (
	rows                           = 300                                // #rows in grid
	columns                        = 300                                // #columns in grid
	block                          = 512                                // size of buf1 and buf2, chunks of data to process
	tmpltime                       = "templates/plottimedata.html"      // html template address
	tmplfrequency                  = "templates/plotfrequencydata.html" // html template address
	tmplplotresponse               = "templates/plotresponse.html"      // html template address
	tmplLMSpredictorALE            = "templates/LMSpredictorALE.html"   // html template address
	tmplfiltersignal               = "templates/filtersignal.html"      // html template address
	addr                           = "127.0.0.1:8080"                   // http server listen address
	patternFilterSignal            = "/filtersignal"                    // http handler pattern for filtering using the ALE predictor
	patternLmsPredictorAle         = "/lmspredictorale"                 // http handler pattern for creating ALE predictor
	patternPlotResponse            = "/plotresponse"                    // http handler for plotting impulse or frequency responses
	xlabels                        = 11                                 // # labels on x axis
	ylabels                        = 11                                 // # labels on y axis
	dataDir                        = "data/"                            // directory for the data files
	deg2rad                        = math.Pi / 180.0                    // convert degrees to radians
	stateFile                      = "filterstate.txt"                  // last M-1 filtered outputs from prev
	stateFileLMS                   = "filterstateLMS.txt"               // last M-1 filtered outputs from prev block
	twoPi                  float64 = 2.0 * math.Pi                      // ampl & freq of signal, time of last sample
	signal                         = "signal.txt"                       // signal to plot
	lmsdesired                     = "lmsdesiredsig.txt"                // LMS desired input consisting of signal and noise
	lmsFilterAle                   = "lmsfilterale.txt"                 // LMS adaptive filter for ALE predictor
)

type Attribute struct {
	Freq     string
	Ampl     string
	FreqName string
	AmplName string
}

// Type to contain all the HTML template actions
type PlotT struct {
	Grid        []string    // plotting grid
	Status      string      // status of the plot
	Xlabel      []string    // x-axis labels
	Ylabel      []string    // y-axis labels
	Filename    string      // filename to plot
	SampleFreq  string      // data sampling rate in Hz
	FFTSegments string      // FFT segments, K
	FFTSize     string      // FFT size
	Samples     string      // complex samples in data file
	SNR         string      // signal-to-noise ratio
	Sines       []Attribute // frequency and amplitude of the sines
}

// Type to hold the minimum and maximum data values
type Endpoints struct {
	xmin float64
	xmax float64
	ymin float64
	ymax float64
}

// Window function type
type Window func(n int, m int) complex128

// properties of the sinusoids for generating the signal
type Sine struct {
	freq int
	ampl float64
}

// previous sample block properties used for generating/filtering current block
type FilterState struct {
	lastFiltered    []float64 // last M-1 incomplete filtered samples from previous block
	firstSampleTime float64   // start time of current submit
	lastSampleTime  float64   // end time of currrent submit
}

type FilterSignal struct {
	sema1            chan int // semaphores to synchronize access to the ping-pong buffers
	sema2            chan int
	wg               sync.WaitGroup
	buf              [][]float64   // ping-pong buffer
	done             chan struct{} // generator Signal to the filter when all samples generated
	samples          int           // total number of samples per submit
	samplesGenerated int           // number of samples generated so far for this submit
	sampleFreq       int           // sample frequency in Hz
	snr              int           // signal to noise ratio in dB
	sines            []Sine        // sinusoids to generate in the signalsignal
	FilterState                    // used by current sample block from previous sample block
	filterCoeff      []float64     // filter coefficients
	filterfile       string        // name of the FIR filter file
	Endpoints                      // embedded struct
}

// LMS algorithm data
type LMSAlgorithm struct {
	delay            int       // reference input delay in samples
	gain             float64   // gain of filter
	trials           int       // number of trials for algorithm
	order            int       // adaptive filter order
	samples          int       // number of samples
	samplesGenerated int       // number of samples generated so far for this trial
	samplerate       int       // sample rate in Hz
	wEnsemble        []float64 // ensemble average coefficients
	wTrial           []float64 // trial coefficients
	wg               sync.WaitGroup
	sema1            chan int // semaphores to synchronize access to the ping-pong buffers
	sema2            chan int
	done             chan struct{} // producer Signal to the consumer when all samples generated
	buf              [][]float64   // ping-pong buffer for filtering buffer
	lastFiltered     []float64     // last M-1 incomplete filtered samples from previous block
	normalizer       float64       // average squared input for normalized LMS
	prevBlockIn      []float64     // last order/2 inputs from previous block
}

var (
	timeTmpl            *template.Template
	freqTmpl            *template.Template
	lmspredictoraleTmpl *template.Template
	plotresponseTmpl    *template.Template
	filterSignalTmpl    *template.Template
	winType             = []string{"Bartlett", "Welch", "Hamming", "Hanning"}
)

// Bartlett window
func bartlett(n int, m int) complex128 {
	real := 1.0 - math.Abs((float64(n)-float64(m))/float64(m))
	return complex(real, 0)
}

// Welch window
func welch(n int, m int) complex128 {
	x := math.Abs((float64(n) - float64(m)) / float64(m))
	real := 1.0 - x*x
	return complex(real, 0)
}

// Hamming window
func hamming(n int, m int) complex128 {
	return complex(.54-.46*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Hanning window
func hanning(n int, m int) complex128 {
	return complex(.5-.5*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Rectangle window
func rectangle(n int, m int) complex128 {
	return 1.0
}

// init parses the html template files and is done only once at startup
func init() {
	timeTmpl = template.Must(template.ParseFiles(tmpltime))
	freqTmpl = template.Must(template.ParseFiles(tmplfrequency))
	plotresponseTmpl = template.Must(template.ParseFiles(tmplplotresponse))
	lmspredictoraleTmpl = template.Must(template.ParseFiles(tmplLMSpredictorALE))
	filterSignalTmpl = template.Must(template.ParseFiles(tmplfiltersignal))
}

// findEndpoints finds the minimum and maximum data values
func (ep *Endpoints) findEndpoints(input *bufio.Scanner, rad float64) {
	ep.xmax = -math.MaxFloat64
	ep.xmin = math.MaxFloat64
	ep.ymax = -math.MaxFloat64
	ep.ymin = math.MaxFloat64
	var (
		n      int = 0 // impulse response plot
		values []string
	)
	for input.Scan() {
		line := input.Text()
		// Each line has 1, 2 or 3 space-separated values, depending on if it is real or complex data:
		// real
		// time real
		// time real imaginary
		values = strings.Split(line, " ")
		var (
			x, y, t float64
			err     error
		)
		// no time data, just real value, used for impulse response plot
		if len(values) == 1 {
			if y, err = strconv.ParseFloat(values[0], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
				continue
			}
			n++
			// real data
		} else if len(values) == 2 {
			if x, err = strconv.ParseFloat(values[0], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
				continue
			}

			if y, err = strconv.ParseFloat(values[1], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[1], err)
				continue
			}
			// complex data
		} else if len(values) == 3 {
			if t, err = strconv.ParseFloat(values[0], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
				continue
			}

			if x, err = strconv.ParseFloat(values[1], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[1], err)
				continue
			}

			if y, err = strconv.ParseFloat(values[2], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[2], err)
				continue
			}

			// Calculate the modulus of the complex data becomes the y-axis
			// The time becomes the x-axis
			y = math.Sqrt(x*x + y*y)
			x = t
		}

		// rotation
		if rad != 0.0 {
			xrot := x*math.Cos(rad) + y*math.Sin(rad)
			yrot := -x*math.Sin(rad) + y*math.Cos(rad)
			x = xrot
			y = yrot
		}

		if x > ep.xmax {
			ep.xmax = x
		}
		if x < ep.xmin {
			ep.xmin = x
		}

		if y > ep.ymax {
			ep.ymax = y
		}
		if y < ep.ymin {
			ep.ymin = y
		}
	}
	// impulse response plot
	if len(values) == 1 {
		ep.xmin = 0.0
		ep.xmax = float64(n - 1)
	}
}

// gridFill inserts the data points in the grid
func gridFill(plot *PlotT, xscale float64, yscale float64, endpoints Endpoints, rad float64, input *bufio.Scanner) error {
	var x float64 = -1
	for input.Scan() {
		line := input.Text()
		// Each line has 1, 2 or 3 space-separated values, depending on if it is real or complex data:
		// real
		// time real
		// time real imaginary
		values := strings.Split(line, " ")
		var (
			y, t float64
			err  error
		)
		// real data, no time
		if len(values) == 1 {
			x++
			if y, err = strconv.ParseFloat(values[0], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
				return err
			}
			// real data
		} else if len(values) == 2 {
			if x, err = strconv.ParseFloat(values[0], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
				return err
			}

			if y, err = strconv.ParseFloat(values[1], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[1], err)
				return err
			}
			// complex data
		} else if len(values) == 3 {
			if t, err = strconv.ParseFloat(values[0], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
				return err
			}

			if x, err = strconv.ParseFloat(values[1], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[1], err)
				return err
			}

			if y, err = strconv.ParseFloat(values[2], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[2], err)
				return err
			}

			// Calculate the modulus of the complex data becomes the y-axis
			// The time becomes the x-axis
			y = math.Sqrt(x*x + y*y)
			x = t
		}

		// rotation
		if rad != 0.0 {
			xrot := x*math.Cos(rad) + y*math.Sin(rad)
			yrot := -x*math.Sin(rad) + y*math.Cos(rad)
			x = xrot
			y = yrot
		}

		// Check if inside the zoom values
		if x < endpoints.xmin || x > endpoints.xmax || y < endpoints.ymin || y > endpoints.ymax {
			continue
		}

		// This cell location (row,col) is on the line
		row := int((endpoints.ymax-y)*yscale + .5)
		col := int((x-endpoints.xmin)*xscale + .5)
		plot.Grid[row*columns+col] = "online"
	}
	return nil
}

// gridFillInterp inserts the data points in the grid and draws a straight line between points
func gridFillInterp(plot *PlotT, xscale float64, yscale float64, endpoints Endpoints, rad float64, input *bufio.Scanner) error {

	var (
		x, y, t      float64
		prevX, prevY float64
		prevSet      bool = true
		err          error
	)

	const lessen = 1
	const increase = 10

	// Get first sample
	input.Scan()
	line := input.Text()
	// Each line has 1, 2 or 3 space-separated values, depending on if it is real or complex data:
	// real
	// time real
	// time real imaginary
	values := strings.Split(line, " ")
	// real data, no time
	if len(values) == 1 {
		x = 0
		if y, err = strconv.ParseFloat(values[0], 64); err != nil {
			fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
			return err
		}
		// real data
	} else if len(values) == 2 {
		if x, err = strconv.ParseFloat(values[0], 64); err != nil {
			fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
			return err
		}

		if y, err = strconv.ParseFloat(values[1], 64); err != nil {
			fmt.Printf("String %s conversion to float error: %v\n", values[1], err)
			return err
		}
		// complex data
	} else if len(values) == 3 {
		if t, err = strconv.ParseFloat(values[0], 64); err != nil {
			fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
			return err
		}

		if x, err = strconv.ParseFloat(values[1], 64); err != nil {
			fmt.Printf("String %s conversion to float error: %v\n", values[1], err)
			return err
		}

		if y, err = strconv.ParseFloat(values[2], 64); err != nil {
			fmt.Printf("String %s conversion to float error: %v\n", values[2], err)
			return err
		}

		// Calculate the modulus of the complex data becomes the y-axis
		// The time becomes the x-axis
		y = math.Sqrt(x*x + y*y)
		x = t
	}

	// rotation
	if rad != 0.0 {
		xrot := x*math.Cos(rad) + y*math.Sin(rad)
		yrot := -x*math.Sin(rad) + y*math.Cos(rad)
		x = xrot
		y = yrot
	}

	// Check if inside the zoom values
	if x < endpoints.xmin || x > endpoints.xmax || y < endpoints.ymin || y > endpoints.ymax {
		prevSet = false
	} else {
		// This cell location (row,col) is on the line
		row := int((endpoints.ymax-y)*yscale + .5)
		col := int((x-endpoints.xmin)*xscale + .5)
		plot.Grid[row*columns+col] = "online"

		prevX = x
		prevY = y
	}

	// Scale factor to determine the number of interpolation points
	lenEP := math.Sqrt((endpoints.xmax-endpoints.xmin)*(endpoints.xmax-endpoints.xmin) +
		(endpoints.ymax-endpoints.ymin)*(endpoints.ymax-endpoints.ymin))

	// Continue with the rest of the points in the file
	for input.Scan() {
		line = input.Text()
		// Each line has 1, 2 or 3 space-separated values, depending on if it is real or complex data:
		// real
		// time real
		// time real imaginary
		values := strings.Split(line, " ")
		// real data, no time
		if len(values) == 1 {
			x++
			if y, err = strconv.ParseFloat(values[0], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
				return err
			}
			// real data
		} else if len(values) == 2 {
			if x, err = strconv.ParseFloat(values[0], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
				return err
			}

			if y, err = strconv.ParseFloat(values[1], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[1], err)
				return err
			}
			// complex data
		} else if len(values) == 3 {
			if t, err = strconv.ParseFloat(values[0], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
				return err
			}

			if x, err = strconv.ParseFloat(values[1], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[1], err)
				return err
			}

			if y, err = strconv.ParseFloat(values[2], 64); err != nil {
				fmt.Printf("String %s conversion to float error: %v\n", values[2], err)
				return err
			}

			// Calculate the modulus of the complex data becomes the y-axis
			// The time becomes the x-axis
			y = math.Sqrt(x*x + y*y)
			x = t
		}

		// rotation
		if rad != 0.0 {
			xrot := x*math.Cos(rad) + y*math.Sin(rad)
			yrot := -x*math.Sin(rad) + y*math.Cos(rad)
			x = xrot
			y = yrot
		}

		// Check if inside the zoom values
		if x < endpoints.xmin || x > endpoints.xmax || y < endpoints.ymin || y > endpoints.ymax {
			continue
		} else if !prevSet {
			prevSet = true
			prevX = x
			prevY = y
		}

		// This cell location (row,col) is on the line
		row := int((endpoints.ymax-y)*yscale + .5)
		col := int((x-endpoints.xmin)*xscale + .5)
		plot.Grid[row*columns+col] = "online"

		// Interpolate the points between previous point and current point

		lenEdge := math.Sqrt((x-prevX)*(x-prevX) + (y-prevY)*(y-prevY))
		ncells := increase * int(columns*lenEdge/lenEP) / lessen // number of points to interpolate
		stepX := (x - prevX) / float64(ncells)
		stepY := (y - prevY) / float64(ncells)

		// loop to draw the points
		interpX := prevX
		interpY := prevY
		for i := 0; i < ncells; i++ {
			row := int((endpoints.ymax-interpY)*yscale + .5)
			col := int((interpX-endpoints.xmin)*xscale + .5)
			plot.Grid[row*columns+col] = "online"
			interpX += stepX
			interpY += stepY
		}

		// Update the previous point with the current point
		prevX = x
		prevY = y
	}
	return nil
}

// processTimeDomain plots the time domain data from disk file
func processTimeDomain(w http.ResponseWriter, r *http.Request, filename string) error {

	// main data structure
	var (
		plot      PlotT
		xscale    float64
		yscale    float64
		endpoints Endpoints
	)

	plot.Grid = make([]string, rows*columns)
	plot.Xlabel = make([]string, xlabels)
	plot.Ylabel = make([]string, ylabels)

	// Open file
	f, err := os.Open(filename)
	if err == nil {
		// Mark the data x-y coordinate online at the corresponding
		// grid row/column.
		input := bufio.NewScanner(f)

		// Determine if rotate requested and perform the rotation of x and y
		rotate := r.FormValue("rotate")
		rad := 0.0
		if len(rotate) > 0 {
			deg, err := strconv.ParseFloat(rotate, 64)
			if err != nil {
				plot.Status = "Rotate degree conversion error"
				fmt.Printf("Rotate degree %v conversion error: %v\n", rotate, err)
			} else {
				rad = deg2rad * deg
			}
		}
		endpoints.findEndpoints(input, rad)

		f.Close()
		f, err = os.Open(filename)
		if err == nil {
			defer f.Close()
			input := bufio.NewScanner(f)

			// Determine if zoom requested and validate endpoints
			zoomxstart := r.FormValue("zoomxstart")
			zoomxend := r.FormValue("zoomxend")
			zoomystart := r.FormValue("zoomystart")
			zoomyend := r.FormValue("zoomyend")
			if len(zoomxstart) > 0 && len(zoomxend) > 0 &&
				len(zoomystart) > 0 && len(zoomyend) > 0 {
				x1, err1 := strconv.ParseFloat(zoomxstart, 64)
				x2, err2 := strconv.ParseFloat(zoomxend, 64)
				y1, err3 := strconv.ParseFloat(zoomystart, 64)
				y2, err4 := strconv.ParseFloat(zoomyend, 64)

				if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
					plot.Status = "Zoom x or y values are not numbers."
					fmt.Printf("Zoom error: x start error = %v, x end error = %v\n", err1, err2)
					fmt.Printf("Zoom error: y start error = %v, y end error = %v\n", err3, err4)
				} else {
					if (x1 < endpoints.xmin || x1 > endpoints.xmax) ||
						(x2 < endpoints.xmin || x2 > endpoints.xmax) || (x1 >= x2) {
						plot.Status = "Zoom values are not in x range."
						fmt.Printf("Zoom error: start or end value not in x range.\n")
					} else if (y1 < endpoints.ymin || y1 > endpoints.ymax) ||
						(y2 < endpoints.ymin || y2 > endpoints.ymax) || (y1 >= y2) {
						plot.Status = "Zoom values are not in y range."
						fmt.Printf("Zoom error: start or end value not in y range.\n")
					} else {
						// Valid Zoom endpoints, replace the previous min and max values
						endpoints.xmin = x1
						endpoints.xmax = x2
						endpoints.ymin = y1
						endpoints.ymax = y2
					}
				}
			}

			// Calculate scale factors for x and y
			xscale = (columns - 1) / (endpoints.xmax - endpoints.xmin)
			yscale = (rows - 1) / (endpoints.ymax - endpoints.ymin)

			// Check for interpolation and fill in the grid with the data points
			interp := r.FormValue("interpolate")
			if interp == "interpolate" {
				err = gridFillInterp(&plot, xscale, yscale, endpoints, rad, input)
			} else {
				err = gridFill(&plot, xscale, yscale, endpoints, rad, input)
			}
			if err != nil {
				return err
			}

			// Set plot status if no errors
			if len(plot.Status) == 0 {
				plot.Status = fmt.Sprintf("Status: Data plotted from (%.3f,%.3f) to (%.3f,%.3f)",
					endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
			}

		} else {
			// Set plot status
			fmt.Printf("Error opening file %s: %v\n", filename, err)
			return fmt.Errorf("error opening file %s: %v", filename, err)
		}
	} else {
		// Set plot status
		fmt.Printf("Error opening file %s: %v\n", filename, err)
		return fmt.Errorf("error opening file %s: %v", filename, err)
	}

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / (xlabels - 1)
	x := endpoints.xmin
	// First label is empty for alignment purposes
	for i := range plot.Xlabel {
		plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	for i := range plot.Ylabel {
		plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}

	// Enter the filename in the form
	plot.Filename = path.Base(filename)

	// Write to HTTP using template and grid
	if err := timeTmpl.Execute(w, plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}

	return nil
}

// processFrequencyDomain calculates the power spectral density (PSD) and plots it
func processFrequencyDomain(w http.ResponseWriter, r *http.Request, filename string) error {
	// Use complex128 for FFT computation
	// Get the number of complex samples nn, open file and count lines, close the file

	var (
		plot          PlotT // main data structure to execute with parsed html template
		endpoints     Endpoints
		N             int                                                        //  complex FFT size
		nn            int                                                        // number of complex samples in the data file
		K             int                                                        //  number of segments used in PSD with 50% overlap
		m             int                                                        // complex segment size
		win           string                                                     // FFT window type
		window        = make(map[string]Window, len(winType))                    // map of window functions
		sumWindow     float64                                                    // sum of squared window values for normalization
		normalizerPSD float64                                                    // normalizer for PSD
		PSD           []float64                                                  // power spectral density
		psdMax        float64                                 = -math.MaxFloat64 // maximum PSD value
		psdMin        float64                                 = math.MaxFloat64  // minimum PSD value
		xscale        float64                                                    // data to grid in x direction
		yscale        float64                                                    // data to grid in y direction
		samplingRate  float64                                                    // sampling rate in Hz
	)

	plot.Grid = make([]string, rows*columns)
	plot.Xlabel = make([]string, xlabels)
	plot.Ylabel = make([]string, ylabels)

	// Put the window functions in the map
	window["Bartlett"] = bartlett
	window["Welch"] = welch
	window["Hamming"] = hamming
	window["Hanning"] = hanning
	window["Rectangle"] = rectangle

	// Open file
	f, err := os.Open(filename)
	if err == nil {
		input := bufio.NewScanner(f)
		// Number of real or complex data samples
		for input.Scan() {
			line := input.Text()
			if len(line) > 0 {
				nn++
			}
		}
		fmt.Printf("Data file %s has %d samples\n", filename, nn)
		// make even number of samples so if segments = 1, we won't
		// do the last FFT with one sample
		if nn%2 == 1 {
			nn++
		}

		f.Close()

		// Get number of segments from HTML form
		// Number of segments to average the periodograms to reduce the variance
		tmp := r.FormValue("fftsegments")
		if len(tmp) == 0 {
			return fmt.Errorf("enter number of FFT segments")
		}
		K, err = strconv.Atoi(tmp)
		if err != nil {
			fmt.Printf("FFT segments string convert error: %v\n", err)
			return fmt.Errorf("fft segments string convert error: %v", err)
		}

		// Require 1 <= K <= 20
		if K < 1 {
			K = 1
		} else if K > 20 {
			K = 20
		}

		// segment size complex samples
		m = nn / (K + 1)

		// Get window type:  Bartlett, Welch, Hanning, Hamming, etc
		// Multiply the samples by the window to reduce spectral leakage
		// caused by high sidelobes in rectangular window
		win = r.FormValue("fftwindow")
		if len(win) == 0 {
			return fmt.Errorf("enter FFT window type")
		}
		w, ok := window[win]
		if !ok {
			fmt.Printf("Invalid FFT window type: %v\n", win)
			return fmt.Errorf("invalid FFT window type: %v", win)
		}
		// sum the window values for PSD normalization due to windowing
		for i := 0; i < 2*m; i++ {
			x := cmplx.Abs(w(i, m))
			sumWindow += x * x
		}
		fmt.Printf("%s window sum = %.2f\n", win, sumWindow)

		// Get FFT size from HTML form
		// Check FFT Size >= 2*m, using 50%  overlap of segments
		// Check FFT Size is a power of 2:  2^n
		tmp = r.FormValue("fftsize")
		if len(tmp) == 0 {
			return fmt.Errorf("enter FFT size")
		}
		N, err = strconv.Atoi(tmp)
		if err != nil {
			return fmt.Errorf("fft size string convert error: %v", err)
		}

		if N < rows {
			fmt.Printf("FFT size < %d\n", rows)
			N = rows
		} else if N > rows*rows {
			fmt.Printf("FFT size > %d\n", rows*rows)
			N = rows * rows
		}
		// This rounds up to nearest FFT size that is a power of 2
		N = int(math.Exp2(float64(int(math.Log2(float64(N)) + .5))))
		fmt.Printf("N=%v\n", N)

		if N < 2*m {
			fmt.Printf("FFT Size %d not greater than 2%d\n", N, 2*m)
			return fmt.Errorf("fft Size %d not greater than 2*%d", N, 2*m)
		}

		// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
		// It is the (sampling frequency)/2, the highest non-aliased frequency
		PSD = make([]float64, N/2+1)

		// Reopen the data file
		f, err = os.Open(filename)
		if err == nil {
			defer f.Close()
			bufm := make([]complex128, m)
			bufN := make([]complex128, N)
			input := bufio.NewScanner(f)
			// Read in initial m samples into buf[m] to start the processing loop
			diff := 0.0
			prev := 0.0
			for k := 0; k < m; k++ {
				input.Scan()
				line := input.Text()
				// Each line has 1, 2 or 3 space-separated values, depending on if it is real or complex data:
				// real
				// time real
				// time real imaginary
				values := strings.Split(line, " ")
				if len(values) == 1 {
					var r float64
					if r, err = strconv.ParseFloat(values[0], 64); err != nil {
						fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
						continue
					}
					if k != 0 {
						diff += .5
					}
					bufm[k] = complex(r, 0)
					// real data
				}
				if len(values) == 2 {
					// time real, calculate the sampling rate from the time steps
					var t, r float64

					if t, err = strconv.ParseFloat(values[0], 64); err != nil {
						fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
						continue
					}
					if r, err = strconv.ParseFloat(values[1], 64); err != nil {
						fmt.Printf("String %s conversion to float error: %v\n", values[1], err)
						continue
					}

					if k == 0 {
						prev = t
					} else {
						diff += t - prev
						prev = t
					}
					bufm[k] = complex(r, 0)

					// complex data
				} else if len(values) == 3 {
					// time real imag
					var t, r, i float64

					// calculate the sampling rate from the time steps
					if t, err = strconv.ParseFloat(values[0], 64); err != nil {
						fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
						continue
					}
					if r, err = strconv.ParseFloat(values[1], 64); err != nil {
						fmt.Printf("String %s conversion to float error: %v\n", values[1], err)
						continue
					}

					if i, err = strconv.ParseFloat(values[2], 64); err != nil {
						fmt.Printf("String %s conversion to float error: %v\n", values[2], err)
						continue
					}

					if k == 0 {
						prev = t
					} else {
						diff += t - prev
						prev = t
					}
					bufm[k] = complex(r, i)
				}
			}

			// Average the time steps and invert to get the sampling rate
			samplingRate = 1.0 / (diff / float64(m-1))
			fmt.Printf("sampling rate = %.2f\n", samplingRate)

			scanOK := true
			// loop over the rest of the file, reading in m samples at a time until EOF
			for {
				// Put the previous m samples in the front of the buffer N to
				// overlap segments
				copy(bufN, bufm)
				// Put the next m samples in back of the previous m samples
				kk := 0
				for k := 0; k < m; k++ {
					scanOK = input.Scan()
					if !scanOK {
						break
					}
					line := input.Text()
					// Each line has 1 - 3 values: [time], real, [imag]
					values := strings.Split(line, " ")
					if len(values) == 1 {
						var r float64

						if r, err = strconv.ParseFloat(values[0], 64); err != nil {
							fmt.Printf("String %s conversion to float error: %v\n", values[0], err)
							continue
						}
						// real data, no time or imag
						bufm[k] = complex(r, 0)
						// real data
					} else if len(values) == 2 {
						// time real, but don't need the time
						var r float64

						if r, err = strconv.ParseFloat(values[1], 64); err != nil {
							fmt.Printf("String %s conversion to float error: %v\n", values[1], err)
							continue
						}
						// real data, imaginary component is zero
						bufm[k] = complex(r, 0)

						// complex data
					} else if len(values) == 3 {
						// time real imag
						var r, i float64
						// Don't need the time
						if r, err = strconv.ParseFloat(values[1], 64); err != nil {
							fmt.Printf("String %s conversion to float error: %v\n", values[1], err)
							continue
						}

						if i, err = strconv.ParseFloat(values[2], 64); err != nil {
							fmt.Printf("String %s conversion to float error: %v\n", values[2], err)
							continue
						}

						bufm[k] = complex(r, i)
					}
					kk++
				}
				// Check for the normal EOF and the abnormal scan error
				// EOF does not give an error but is considered normal termination
				if !scanOK {
					if input.Err() != nil {
						fmt.Printf("Data file scan error: %v\n", input.Err().Error())
						return fmt.Errorf("data file scan error: %v", input.Err())
					}
				}
				// Put the next kk samples in back of the previous m
				copy(bufN[m:], bufm[:kk])

				// window the m + kk samples with chosen window
				for i := 0; i < m+kk; i++ {
					bufN[i] *= w(i, m)
				}

				// zero-pad N-m-kk samples in buf[N]
				for i := m + kk; i < N; i++ {
					bufN[i] = 0
				}

				// Perform N-point complex FFT and add squares to previous values in PSD[N/2+1]
				fourierN := fft.FFT(bufN)
				x := cmplx.Abs(fourierN[0])
				PSD[0] += x * x
				for i := 1; i < N/2; i++ {
					// Use positive and negative frequencies -> bufN[N-i] = bufN[-i]
					xi := cmplx.Abs(fourierN[i])
					xNi := cmplx.Abs(fourierN[N-i])
					PSD[i] += xi*xi + xNi*xNi
				}
				x = cmplx.Abs(fourierN[N/2])
				PSD[N/2] += x * x

				// part of K*Sum(w[i]*w[i]) PSD normalizer
				normalizerPSD += sumWindow

				// EOF reached
				if !scanOK {
					break
				}
			} // K segments done

			// Normalize the PSD using K*Sum(w[i]*w[i])
			// Use log plot for wide dynamic range
			if r.FormValue("plottype") == "linear" {
				for i := range PSD {
					PSD[i] /= normalizerPSD
					if PSD[i] > psdMax {
						psdMax = PSD[i]
					}
					if PSD[i] < psdMin {
						psdMin = PSD[i]
					}
				}
				// log10 in dB
			} else {
				for i := range PSD {
					PSD[i] /= normalizerPSD
					PSD[i] = 10.0 * math.Log10(PSD[i])
					if PSD[i] > psdMax {
						psdMax = PSD[i]
					}
					if PSD[i] < psdMin {
						psdMin = PSD[i]
					}
				}
			}

			endpoints.xmin = 0
			endpoints.xmax = float64(N / 2) // equivalent to Nyquist critical frequency
			endpoints.ymin = psdMin
			endpoints.ymax = psdMax

			// Calculate scale factors for x and y
			xscale = (columns - 1) / (endpoints.xmax - endpoints.xmin)
			yscale = (rows - 1) / (endpoints.ymax - endpoints.ymin)

			// Store the PSD in the plot Grid
			for bin, pow := range PSD {
				row := int((endpoints.ymax-pow)*yscale + .5)
				col := int((float64(bin)-endpoints.xmin)*xscale + .5)
				plot.Grid[row*columns+col] = "online"
			}

			// Store in the form:  FFT Size, window type, number of samples nn, K segments, sampling frequency
			// Plot the PSD N/2 float64 values, execute the data on the plotfrequency.html template

			// Set plot status if no errors
			if len(plot.Status) == 0 {
				plot.Status = fmt.Sprintf("Status: Data plotted from (%.3f,%.3f) to (%.3f,%.3f)",
					endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
			}

		} else {
			// Set plot status
			fmt.Printf("Error opening file %s: %v\n", filename, err)
			return fmt.Errorf("error opening file %s: %v", filename, err)
		}
	} else {
		// Set plot status
		fmt.Printf("Error opening file %s: %v\n", filename, err)
		return fmt.Errorf("error opening file %s: %v", filename, err)
	}

	// Apply the  sampling rate in Hz to the x-axis using a scale factor
	// Convert the fft size to sampling rate/2, the Nyquist critical frequency
	sf := 0.5 * samplingRate / endpoints.xmax

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / (xlabels - 1)
	format := "%.0f"
	if incr*sf < 1.0 {
		format = "%.2f"
	}
	x := endpoints.xmin
	// First label is empty for alignment purposes
	for i := range plot.Xlabel {
		plot.Xlabel[i] = fmt.Sprintf(format, x*sf)
		x += incr
	}

	// Construct the y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	for i := range plot.Ylabel {
		plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}

	// Insert frequency domain parameters in the form
	plot.SampleFreq = fmt.Sprintf("%.0f", samplingRate)
	plot.FFTSegments = strconv.Itoa(K)
	plot.FFTSize = strconv.Itoa(N)
	plot.Samples = strconv.Itoa(nn)

	// Enter the filename in the form
	plot.Filename = path.Base(filename)

	// Write to HTTP using template and grid
	if err := freqTmpl.Execute(w, plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}

	return nil
}

// handlePlotResponse displays the impulse or frequency response of the
// ALE predictor
func handlePlotResponse(w http.ResponseWriter, r *http.Request) {
	// main data structure
	var (
		plot PlotT
		err  error = nil
	)

	filename := r.FormValue("filename")
	// choose time or frequency domain processing
	if len(filename) > 0 {

		domain := r.FormValue("domain")
		switch domain {
		case "time":
			err = processTimeDomain(w, r, path.Join(dataDir, filename))
			if err != nil {
				plot.Status = err.Error()
			}
		case "frequency":
			err = processFrequencyDomain(w, r, path.Join(dataDir, filename))
			if err != nil {
				plot.Status = err.Error()
			}
		default:
			plot.Status = fmt.Sprintf("Invalid domain choice: %s", domain)
		}

		if err != nil {

			// Write to HTTP using template and grid``
			if err := plotresponseTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
		}
	} else {
		plot.Status = "Missing filename to plot"
		// Write to HTTP using template and grid``
		if err := plotresponseTmpl.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	}
}

// fillBuf populates the buffer with signal samples from disk
func (lms *LMSAlgorithm) fillBuf(n int, input *bufio.Scanner) int {

	// Determine how many samples we need to generate
	howMany := block
	toGo := lms.samples - lms.samplesGenerated
	if toGo < block {
		howMany = toGo
	}

	// Each line has a time(sec) and signal value, both floats.
	// We don't need the time here.
	lms.normalizer = 0
	for i := 0; i < howMany; i++ {
		input.Scan()
		line := input.Text()
		values := strings.Split(line, " ")
		val, err := strconv.ParseFloat(values[1], 64)
		if err != nil {
			fmt.Printf("input sample conversion error: %v\n", err)
			continue
		}
		lms.buf[n][i] = val
		lms.normalizer += val * val
	}

	// Save the samples generated so far
	lms.samplesGenerated += howMany
	// Average the normalizer
	lms.normalizer /= float64(howMany)

	// Save order samples from previous block error calculation.
	// This is used as the desired value in the LMS error calculation
	// and updating the weight.
	n1 := block - lms.order
	n2 := (n + 1) % 2
	j := 0
	for i := n1; i < block; i++ {
		lms.prevBlockIn[j] = lms.buf[n2][i]
		j++
	}
	return howMany
}

// generateLMSData generates a new set of input data and launches a Go routine to read blocks
func (lms *LMSAlgorithm) generateLMSData(w http.ResponseWriter, r *http.Request) error {

	// generate a new set of input data for this trial
	if err := processSignalData(w, r, lms.samples, lms.samplerate); err != nil {
		return fmt.Errorf("processSignalData error: %v", err)
	}
	// fillBuf with x(n)

	f, err := os.Open(path.Join(dataDir, lmsdesired))
	if err != nil {
		log.Fatalf("Error opening %s: %v\n", lmsdesired, err.Error())
	}
	input := bufio.NewScanner(f)

	// increment wg
	lms.wg.Add(1)

	// launch a goroutine to generate input sample blocks
	go func(input *bufio.Scanner) {
		// if all samples generated, signal filter on done semaphore and return
		// close lmsdesired file and signal handler this iteration is done
		defer func() {
			lms.done <- struct{}{}
			f.Close()
			lms.wg.Done()
		}()

		// loop to generate a block of signal samples
		// signal the runLMS when done with each block of samples
		// block on a semaphore until runLMS goroutine is available
		for {
			n := lms.fillBuf(0, input)
			lms.sema1 <- n
			if n < block {
				return
			}
			n = lms.fillBuf(1, input)
			lms.sema2 <- n
			if n < block {
				return
			}
		}
	}(input)

	return nil
}

// generateSinsuoids creates a sum of sine waves in Gaussian noise at SNR
func generateSinusoids(w http.ResponseWriter, r *http.Request, samples int, samplerate int) error {
	// get snr
	temp := r.FormValue("SSsnr")
	if len(temp) == 0 {
		return fmt.Errorf("missing SNR for Sum of Sinsuoids")
	}
	snr, err := strconv.Atoi(temp)
	if err != nil {
		return err
	}

	type Sine struct {
		freq int
		amp  int
	}

	var (
		maxampl  int      = 0 // maximum sine amplitude
		noiseSD  float64      // noise standard deviation
		freqName []string = []string{"SSfreq1", "SSfreq2",
			"SSfreq3", "SSfreq4", "SSfreq5"}
		ampName []string = []string{"SSamp1", "SSamp2", "SSamp3",
			"SSamp4", "SSamp5"}
		signal []Sine = make([]Sine, 0) // sines to generate
	)

	// get the sine frequencies and amplitudes, 1 to 5 possible
	for i, name := range freqName {
		a := r.FormValue(ampName[i])
		f := r.FormValue(name)
		if len(a) > 0 && len(f) > 0 {
			freq, err := strconv.Atoi(f)
			if err != nil {
				return err
			}
			amp, err := strconv.Atoi(a)
			if err != nil {
				return err
			}
			signal = append(signal, Sine{freq: freq, amp: amp})
			if amp > maxampl {
				maxampl = amp
			}
		}
	}
	// Require at least one sine to create
	if len(signal) == 0 {
		return fmt.Errorf("enter frequency and amplitude of 1 to 5 sinsuoids")
	}

	// Calculate the noise standard deviation using the SNR and maxampl
	ratio := math.Pow(10.0, float64(snr)/10.0)
	noiseSD = math.Sqrt(0.5 * float64(maxampl) * float64(maxampl) / ratio)
	file, err := os.Create(path.Join(dataDir, lmsdesired))
	if err != nil {
		return fmt.Errorf("create %v error: %v", path.Join(dataDir, lmsdesired), err)
	}
	defer file.Close()

	// Sum the sinsuoids and noise over the time interval
	delta := 1.0 / float64(samplerate)
	twoPi := 2.0 * math.Pi
	for t := 0.0; t < float64(samples)*delta; t += delta {
		sinesum := 0.0
		for _, sig := range signal {
			omega := twoPi * float64(sig.freq)
			sinesum += float64(sig.amp) * math.Sin(omega*t)
		}
		sinesum += noiseSD * rand.NormFloat64()
		fmt.Fprintf(file, "%v %v\n", t, sinesum)
	}

	return nil
}

// generateAM creates an amplitude modulation waveform
func generateAM(w http.ResponseWriter, r *http.Request, samples int, samplerate int) error {
	// get modulation percent
	temp := r.FormValue("percentmodulation")
	if len(temp) == 0 {
		return fmt.Errorf("missing Modulation Percent for AM")
	}
	percentModulation, err := strconv.Atoi(temp)
	if err != nil {
		return err
	}

	// get carrier frequency
	temp = r.FormValue("carrierfreq")
	if len(temp) == 0 {
		return fmt.Errorf("missing Carrier Frequency  for AM")
	}
	f1, err := strconv.Atoi(temp)
	if err != nil {
		return err
	}

	// get modulating frequency
	temp = r.FormValue("modulfreq")
	if len(temp) == 0 {
		return fmt.Errorf("missing Modulating Frequency for AM")
	}
	f2, err := strconv.Atoi(temp)
	if err != nil {
		return err
	}

	// alias check
	if f1-f2 < 0 {
		return fmt.Errorf("aliasing:  carrier frequency - modulating frequency < 0")
	}
	if f1+f2 > samplerate/2 {
		return fmt.Errorf("aliasing:  carrier frequency + modulating frequency > samplerate/2")
	}

	file, err := os.Create(path.Join(dataDir, lmsdesired))
	if err != nil {
		return fmt.Errorf("create %v error: %v", path.Join(dataDir, lmsdesired), err)
	}
	defer file.Close()

	// Create the AM waveform using sin(2*PI*f1*t)*(1 + A*cos(2*PI*f2*t))
	// where f1 is the carrier frequency, f2 is the modulating frequency,
	// and A is the percent modulation/100
	delta := 1.0 / float64(samplerate)
	const twoPi = 2.0 * math.Pi
	A := float64(percentModulation) / 100.0
	for t := 0.0; t < float64(samples)*delta; t += delta {
		y1 := math.Sin(twoPi * float64(f1) * t)
		y2 := 1.0 + A*math.Cos(twoPi*float64(f2)*t)
		fmt.Fprintf(file, "%v %v\n", t, y1*y2)
	}

	return nil
}

// generateFM creates a frequency modulation waveform, either sine baseband or linear (LFM)
func generateFM(w http.ResponseWriter, r *http.Request, samples int, samplerate int) error {

	generateLFM := func() error {
		// get bandwidth
		temp := r.FormValue("FMbandwidth")
		if len(temp) == 0 {
			return fmt.Errorf("missing Bandwidth  for FM linear")
		}
		bw, err := strconv.Atoi(temp)
		if err != nil {
			return err
		}

		// get center frequency
		temp = r.FormValue("FMfrequency")
		if len(temp) == 0 {
			return fmt.Errorf("missing Center Frequency for FM linear")
		}
		fc, err := strconv.Atoi(temp)
		if err != nil {
			return err
		}

		// check for aliasing
		if fc-bw/2 < 0 {
			return fmt.Errorf("aliasing:  fc-bw/2 < 0")
		}
		if fc+bw/2 > samplerate/2 {
			return fmt.Errorf("aliasing: fc+bw/2 > samplerate/2")
		}

		file, err := os.Create(path.Join(dataDir, lmsdesired))
		if err != nil {
			return fmt.Errorf("create %v error: %v", path.Join(dataDir, lmsdesired), err)
		}
		defer file.Close()

		// Create the FM waveform using A*cos(2*PI*(fc-bw/2)*t + PI*(bw/tau)*t*t)
		// where fc is the center frequency, bw is the bandwidth, tau is the pulse
		// duration in seconds
		delta := 1.0 / float64(samplerate)
		tau := delta * float64(samples)
		const (
			pi    = math.Pi
			twoPi = 2.0 * pi
		)
		A := 1.0
		for t := 0.0; t < tau; t += delta {
			ang := twoPi*(float64(fc)-float64(bw)/2.0)*t + pi*(float64(bw)/tau)*t*t
			fmt.Fprintf(file, "%v %v\n", t, A*math.Cos(ang))
		}
		return nil
	}

	generateSineFM := func() error {
		// get center frequency
		temp := r.FormValue("FMfrequency")
		if len(temp) == 0 {
			return fmt.Errorf("missing Center Frequency for FM sine")
		}
		fc, err := strconv.Atoi(temp)
		if err != nil {
			return err
		}

		// get frequency deviation
		temp = r.FormValue("FMfreqdev")
		if len(temp) == 0 {
			return fmt.Errorf("missing frequency deviation for FM sine")
		}
		fd, err := strconv.Atoi(temp)
		if err != nil {
			return err
		}

		// get modulating frequency
		temp = r.FormValue("FMmodfreq")
		if len(temp) == 0 {
			return fmt.Errorf("missing modulating frequency for FM sine")
		}
		fm, err := strconv.Atoi(temp)
		if err != nil {
			return err
		}

		// check for aliasing
		if fc-fd < 0 {
			return fmt.Errorf("aliasing:  fc-bw/2 < 0")
		}
		if fc+fd > samplerate/2 {
			return fmt.Errorf("aliasing: fc+bw/2 > samplerate/2")
		}

		file, err := os.Create(path.Join(dataDir, lmsdesired))
		if err != nil {
			return fmt.Errorf("create %v error: %v", path.Join(dataDir, lmsdesired), err)
		}
		defer file.Close()

		// Create the FM waveform using A*cos(2*PI*fc*t + (fd/fm)*sin(2*PI*fm*t))
		// where fc is the center frequency, fd is the frequency deviation, fm is
		// the modulating frequency.
		delta := 1.0 / float64(samplerate)
		tau := delta * float64(samples)
		const (
			twoPi = 2.0 * math.Pi
		)
		A := 1.0
		for t := 0.0; t < tau; t += delta {
			y := A * math.Cos(twoPi*float64(fc)*t+
				float64(fd)/float64(fm)*math.Sin(twoPi*float64(fm)*t))
			fmt.Fprintf(file, "%v %v\n", t, y)
		}

		return nil
	}

	// Determine if Sine baseband or linear (LFM) and create it
	temp := r.FormValue("FMtype")
	switch temp {
	case "linear":
		return generateLFM()
	case "sine":
		return generateSineFM()
	}
	return nil
}

// processSignalData generates and saves to disk the desired signal
func processSignalData(w http.ResponseWriter, r *http.Request, samples int, samplerate int) error {

	// Determine which signal to generate
	switch r.FormValue("signaltype") {
	case "sumsinusoids":
		err := generateSinusoids(w, r, samples, samplerate)
		if err != nil {
			return err
		}
	case "am":
		err := generateAM(w, r, samples, samplerate)
		if err != nil {
			return err
		}
	case "fm":
		err := generateFM(w, r, samples, samplerate)
		if err != nil {
			return err
		}
	}
	return nil
}

// handleLmsPredictorAle creates an ALE predictor using the LMS algorithm
func handleLmsPredictorAle(w http.ResponseWriter, r *http.Request) {
	var plot PlotT

	// Get the number of samples to generate and the sample rate
	// The number of samples determines the number of iterations of the LMS algorithm
	samplestxt := r.FormValue("samples")
	sampleratetxt := r.FormValue("samplerate")
	// choose time or frequency domain processing
	if len(samplestxt) > 0 && len(sampleratetxt) > 0 {

		samples, err := strconv.Atoi(samplestxt)
		if err != nil {
			plot.Status = fmt.Sprintf("Samples conversion to int error: %v", err.Error())
			fmt.Printf("Samples conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := lmspredictoraleTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		samplerate, err := strconv.Atoi(sampleratetxt)
		if err != nil {
			plot.Status = fmt.Sprintf("Sample rate conversion to int error: %v", err.Error())
			fmt.Printf("Sample rate conversion to int error: %v\n", err.Error())
			// Write to HTTP using template and grid
			if err := lmspredictoraleTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// adaptive filter order is the number of past samples
		// filter length = order + 1
		txt := r.FormValue("filtorder")
		order, err := strconv.Atoi((txt))
		if err != nil {
			plot.Status = fmt.Sprintf("Filter order conversion to int error: %v", err.Error())
			fmt.Printf("Filter order conversion to int error: %v\n", err.Error())
			// Write to HTTP using template and grid
			if err := lmspredictoraleTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		// Make filter length odd, which means filter order is even;
		// if order is 7, it is changed to 8, and the length becomes 9.
		order = (order + 1) / 2 * 2

		// reference input delay in samples to decorrelate the noise
		txt = r.FormValue("delay")
		delay, err := strconv.Atoi((txt))
		if err != nil {
			plot.Status = fmt.Sprintf("Delay conversion to int error: %v", err.Error())
			fmt.Printf("Delay conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := lmspredictoraleTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// delay restriction
		if delay > order/2 {
			plot.Status = "delay must be less than order/2"
			fmt.Printf("Delay must be less than order/2\n")
			// Write to HTTP using template and grid
			if err := lmspredictoraleTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// gain factor mu that regulates the speed and stability of adaption
		txt = r.FormValue("gain")
		gain, err := strconv.ParseFloat(txt, 64)
		if err != nil {
			plot.Status = fmt.Sprintf("Gain conversion to float64 error: %v", err.Error())
			fmt.Printf("Gain conversion to float64 error: %v\n", err.Error())
			// Write to HTTP using template and grid
			if err := lmspredictoraleTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// number of trials to perform the LMS algorithm to get ensemble average of the
		// weights
		txt = r.FormValue("trials")
		trials, err := strconv.Atoi((txt))
		if err != nil {
			plot.Status = fmt.Sprintf("Trials conversion to int error: %v", err.Error())
			fmt.Printf("Trials conversion to int error: %v\n", err.Error())
			// Write to HTTP using template and grid
			if err := lmspredictoraleTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Construct object to hold LMS algorithm parameters
		lmsPredictor := LMSAlgorithm{
			delay:        delay,
			gain:         gain,
			trials:       trials,
			order:        order,
			samples:      samples,
			samplerate:   samplerate,
			wEnsemble:    make([]float64, order+1),
			wTrial:       make([]float64, order+1),
			sema1:        make(chan int),
			sema2:        make(chan int),
			buf:          make([][]float64, 2),
			done:         make(chan struct{}),
			lastFiltered: make([]float64, order),
			prevBlockIn:  make([]float64, order),
		}
		lmsPredictor.buf[0] = make([]float64, block)
		lmsPredictor.buf[1] = make([]float64, block)

		// Run the Least-Mean-Square (LMS) algorithm to create the adaptive filter
		// Loop over the trials to generate the ensemble of filters which is averaged.
		for i := 0; i < lmsPredictor.trials; i++ {
			// Generate a new set of input data
			err = lmsPredictor.generateLMSData(w, r)
			if err != nil {
				plot.Status = fmt.Sprintf("generateLMSData error: %v", err.Error())
				fmt.Printf("generateLMSData error: %v\n", err.Error())
				// Write to HTTP using template and grid
				if err := lmspredictoraleTmpl.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}

			// Run the LMS algorithm to find the adaptive filter coefficients
			err = lmsPredictor.runLms()
			if err != nil {
				plot.Status = fmt.Sprintf("generateLMSData error: %v", err.Error())
				fmt.Printf("runLmsPredictor error: %v\n", err.Error())
				// Write to HTTP using template and grid
				if err := lmspredictoraleTmpl.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			// Wait for this trial to finish
			lmsPredictor.wg.Wait()
		}

		// Save ensemble filter coefficients to disk (averaged coefficients)
		file, err := os.Create(path.Join(dataDir, lmsFilterAle))
		if err != nil {
			plot.Status = fmt.Sprintf("create %v error: %v", path.Join(dataDir, lmsFilterAle), err.Error())
			fmt.Printf("create %v error: %v\n", path.Join(dataDir, lmsFilterAle), err.Error())
			// Write to HTTP using template and grid
			if err := lmspredictoraleTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		defer file.Close()

		// Average the weight ensemble to reduce variance
		for i := 0; i < lmsPredictor.order+1; i++ {
			fmt.Fprintf(file, "%v\n", lmsPredictor.wEnsemble[i]/float64(trials))
		}

		plot.Status = fmt.Sprintf("Adaptive filter weights '%s' written to the data folder", lmsFilterAle)
		// Write to HTTP using template and grid
		if err := lmspredictoraleTmpl.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}

	} else {
		plot.Status = "Enter number of samples and sample rate"
		// Write to HTTP using template and grid
		if err := lmspredictoraleTmpl.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	}
}

// fillBuf populates the buffer with signal samples, fst = first sample time
func (fs *FilterSignal) fillBuf(n int, noiseSD float64, fst float64) int {
	// get last sample time from previous block
	// fill buf n with noisy sinusoids with given SNR
	// Sum the sinsuoids and noise and insert into the buffer

	// Determine how many samples we need to generate
	howMany := block
	toGo := fs.samples - fs.samplesGenerated
	if toGo < block {
		howMany = toGo
	}

	delta := 1.0 / float64(fs.sampleFreq)
	t := fst
	for i := 0; i < howMany; i++ {
		sinesum := 0.0
		for _, sig := range fs.sines {
			omega := twoPi * float64(sig.freq)
			sinesum += float64(sig.ampl) * math.Sin(omega*t)
		}
		sinesum += noiseSD * rand.NormFloat64()
		fs.buf[n][i] = sinesum
		t += delta
	}
	// Save the next sample time for next block of samples
	fs.lastSampleTime = t
	fs.samplesGenerated += howMany
	return howMany
}

// gridFillInterp inserts the data points in the grid and draws a straight line between points
func (fs *FilterSignal) gridFillInterp(plot *PlotT) error {
	var (
		x            float64 = fs.firstSampleTime
		y            float64 = 0.0
		prevX, prevY float64
		err          error
		xscale       float64
		yscale       float64
		input        *bufio.Scanner
		timeStep     float64 = 1.0 / float64(fs.sampleFreq)
	)

	// Mark the data x-y coordinate online at the corresponding
	// grid row/column.

	fs.xmin = fs.firstSampleTime
	fs.xmax = fs.lastSampleTime

	// Calculate scale factors for x and y
	xscale = (columns - 1) / (fs.xmax - fs.xmin)
	yscale = (rows - 1) / (fs.ymax - fs.ymin)

	f, err := os.Open(path.Join(dataDir, signal))
	if err != nil {
		fmt.Printf("Error opening %s: %v\n", signal, err.Error())
		return err
	}
	defer f.Close()
	input = bufio.NewScanner(f)

	// Get first sample
	input.Scan()
	value := input.Text()

	if y, err = strconv.ParseFloat(value, 64); err != nil {
		fmt.Printf("gridFillInterp first sample string %s conversion to float error: %v\n", value, err)
		return err
	}

	plot.Grid = make([]string, rows*columns)

	// This cell location (row,col) is on the line
	row := int((fs.ymax-y)*yscale + .5)
	col := int((x-fs.xmin)*xscale + .5)
	plot.Grid[row*columns+col] = "online"

	prevX = x
	prevY = y

	// Scale factor to determine the number of interpolation points
	lenEPy := fs.ymax - fs.ymin
	lenEPx := fs.xmax - fs.xmin

	// Continue with the rest of the points in the file
	for input.Scan() {
		x += timeStep
		value = input.Text()
		if y, err = strconv.ParseFloat(value, 64); err != nil {
			fmt.Printf("gridFillInterp the rest of file string %s conversion to float error: %v\n", value, err)
			return err
		}

		// This cell location (row,col) is on the line
		row := int((fs.ymax-y)*yscale + .5)
		col := int((x-fs.xmin)*xscale + .5)
		plot.Grid[row*columns+col] = "online"

		// Interpolate the points between previous point and current point

		/* lenEdge := math.Sqrt((x-prevX)*(x-prevX) + (y-prevY)*(y-prevY)) */
		lenEdgeX := math.Abs((x - prevX))
		lenEdgeY := math.Abs(y - prevY)
		ncellsX := int(columns * lenEdgeX / lenEPx) // number of points to interpolate in x-dim
		ncellsY := int(rows * lenEdgeY / lenEPy)    // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := ncellsX
		if ncellsY > ncells {
			ncells = ncellsY
		}

		stepX := (x - prevX) / float64(ncells)
		stepY := (y - prevY) / float64(ncells)

		// loop to draw the points
		interpX := prevX
		interpY := prevY
		for i := 0; i < ncells; i++ {
			row := int((fs.ymax-interpY)*yscale + .5)
			col := int((interpX-fs.xmin)*xscale + .5)
			plot.Grid[row*columns+col] = "online"
			interpX += stepX
			interpY += stepY
		}

		// Update the previous point with the current point
		prevX = x
		prevY = y
	}
	return nil
}

// filterBufLMS runs the LMS algorithm on the buffer
func (lms *LMSAlgorithm) runLmsBuf(index int, nsamples int) {
	// Have the last partial filter outputs from previous block in LMSAlgorithm.
	// Loop over the samples from the generator and apply the filter coefficients
	// in a convolution sum.  Compute the error from the desired and update the
	// weight.

	// m is the number of coefficients in the adaptive filter
	m := lms.order + 1
	end := m - 1
	if nsamples < m-1 {
		end = nsamples
	}

	// delay of desired from current y, taking into account causality of FIR filter
	d := lms.delay + lms.order/2

	// gain and normalizer
	g := lms.gain / (float64(nsamples) * lms.normalizer)

	// Incorporate the previous block partially filtered
	for n := 0; n < end; n++ {
		sum := lms.lastFiltered[n]
		for k := 0; k <= n; k++ {
			sum += lms.buf[index][k] * lms.wTrial[n-k]
		}
		var err float64 = 0.0
		// inputs from previous block
		dif := n - d
		// error taking into account the delay due to causal fir filter
		if dif < 0 {
			err = lms.prevBlockIn[lms.order+dif] - sum
		} else {
			err = lms.buf[index][dif] - sum
		}
		// update the adaptive weights
		for i := range lms.wTrial {
			dif := n - i
			if dif < 0 {
				lms.wTrial[i] += g * err * lms.prevBlockIn[lms.order+dif]
			} else {
				lms.wTrial[i] += g * err * lms.buf[index][dif]
			}
			lms.wEnsemble[i] += lms.wTrial[i]
		}
	}

	// This is the last block and it has no more samples since nsamples <= m-1
	if end == nsamples {
		return
	}

	for n := end; n < nsamples; n++ {
		sum := 0.0
		for k := n - m + 1; k <= n; k++ {
			sum += lms.buf[index][k] * lms.wTrial[n-k]
		}
		// error taking into account the delay due to causal fir filter
		err := lms.buf[index][n-d] - sum
		// update the adaptive weights
		for i := range lms.wTrial {
			lms.wTrial[i] += g * err * lms.buf[index][n-i]
			lms.wEnsemble[i] += lms.wTrial[i]
		}
	}

	// Generate the partially filtered outputs used in next block
	i := 0
	if nsamples == block {
		for n := block - m + 1; n < block; n++ {
			sum := 0.0
			for k := n; k < block; k++ {
				sum += lms.buf[index][k] * lms.wTrial[m-k+n-1]
			}
			lms.lastFiltered[i] = sum
			i++
		}
	}
}

// filterBuf filters the samples with the given filter coefficients
// index is the buffer to use, 1 or 2, nsamples is the number of samples to filter
func (fs *FilterSignal) filterBuf(index int, nsamples int, f *os.File) {
	// Have the last partial filter outputs from previous block in FilterSignal.
	// Loop over the samples from the generator and apply the filter coefficients
	// in a convolution sum.

	// Incorporate the previous block partially filtered outputs
	m := len(fs.filterCoeff)
	end := m - 1
	if nsamples < m-1 {
		end = nsamples
	}
	for n := 0; n < end; n++ {
		sum := fs.lastFiltered[n]
		for k := 0; k <= n; k++ {
			sum += fs.buf[index][k] * fs.filterCoeff[n-k]
		}
		// find min/max of the signal as we go
		if sum < fs.ymin {
			fs.ymin = sum
		}
		if sum > fs.ymax {
			fs.ymax = sum
		}
		fmt.Fprintf(f, "%f\n", sum)
	}

	// This is the last block and it has no more samples since nsamples <= m-1
	if end == nsamples {
		return
	}

	for n := end; n < nsamples; n++ {
		sum := 0.0
		for k := n - m + 1; k <= n; k++ {
			sum += fs.buf[index][k] * fs.filterCoeff[n-k]
		}
		// find min/max of the signal as we go
		if sum < fs.ymin {
			fs.ymin = sum
		}
		if sum > fs.ymax {
			fs.ymax = sum
		}
		fmt.Fprintf(f, "%f\n", sum)
	}

	// Generate the partially filtered outputs used in next block
	i := 0
	if nsamples == block {
		for n := block - m + 1; n < block; n++ {
			sum := 0.0
			for k := n; k < block; k++ {
				sum += fs.buf[index][k] * fs.filterCoeff[m-k+n-1]
			}
			fs.lastFiltered[i] = sum
			i++
		}
	}
}

// nofilterBuf saves the signal to a file.  It is not modified.
// index is the buffer to use, 1 or 2, nsamples is the number of samples to filter
func (fs *FilterSignal) nofilterBuf(index int, nsamples int, f *os.File) {
	for n := 0; n < nsamples; n++ {
		// find min/max of the signal as we go
		if fs.buf[index][n] < fs.ymin {
			fs.ymin = fs.buf[index][n]
		}
		if fs.buf[index][n] > fs.ymax {
			fs.ymax = fs.buf[index][n]
		}
		fmt.Fprintf(f, "%f\n", fs.buf[index][n])
	}
}

// generate creates the noisy signal, it is the producer or generator
func (fs *FilterSignal) generate(r *http.Request) error {

	// get SNR, sine frequencies and sine amplitudes
	temp := r.FormValue("snr")
	if len(temp) == 0 {
		return fmt.Errorf("missing SNR for Sum of Sinsuoids")
	}
	snr, err := strconv.Atoi(temp)
	if err != nil {
		return err
	}

	fs.snr = snr
	var (
		maxampl  float64  = 0.0 // maximum sine amplitude
		freqName []string = []string{"SSfreq1", "SSfreq2",
			"SSfreq3", "SSfreq4", "SSfreq5"}
		ampName []string = []string{"SSamp1", "SSamp2", "SSamp3",
			"SSamp4", "SSamp5"}
	)

	// get the sine frequencies and amplitudes, 1 to 5 possible
	for i, name := range freqName {
		a := r.FormValue(ampName[i])
		f := r.FormValue(name)
		if len(a) > 0 && len(f) > 0 {
			freq, err := strconv.Atoi(f)
			if err != nil {
				return err
			}
			ampl, err := strconv.ParseFloat(a, 64)
			if err != nil {
				return err
			}
			fs.sines = append(fs.sines, Sine{freq: freq, ampl: ampl})
			if ampl > maxampl {
				maxampl = ampl
			}
		}
	}

	// Require at least one sine to create
	if len(fs.sines) == 0 {
		return fmt.Errorf("enter frequency and amplitude of 1 to 5 sinsuoids")
	}

	// Calculate the noise standard deviation using the SNR and maxampl
	ratio := math.Pow(10.0, float64(snr)/10.0)
	noiseSD := math.Sqrt(0.5 * float64(maxampl) * float64(maxampl) / ratio)

	// increment wg
	fs.wg.Add(1)

	// launch a goroutine to generate samples
	go func() {
		// if all samples generated, signal filter on done semaphore and return
		defer fs.wg.Done()
		defer func() {
			fs.done <- struct{}{}
		}()

		fst := fs.firstSampleTime
		// loop to generate a block of signal samples
		// signal the filter when done with each block of samples
		// block on a semaphore until filter goroutine is available
		// set the first sample time equal to the previous last sample time
		for {
			n := fs.fillBuf(0, noiseSD, fst)
			fst = fs.lastSampleTime
			fs.sema1 <- n
			if n < block {
				return
			}
			n = fs.fillBuf(1, noiseSD, fst)
			fst = fs.lastSampleTime
			fs.sema2 <- n
			if n < block {
				return
			}
		}
	}()

	return nil
}

// runLms updates the adaptive weights using the LMS algorithm
func (lms *LMSAlgorithm) runLms() error {

	// increment wg
	lms.wg.Add(1)

	// launch a goroutine to filter generator signal
	// select on the generator semaphores and the done channel
	go func() {
		defer lms.wg.Done()
		for {
			select {
			case n := <-lms.sema1:
				lms.runLmsBuf(0, n)
			case n := <-lms.sema2:
				lms.runLmsBuf(1, n)
			case <-lms.done:
				// if done signal from generator, save state, then return
				// save the Signal state incomplete filtered output
				f, err := os.Create(path.Join(dataDir, stateFileLMS))
				if err != nil {
					fmt.Printf("Create %s save state error: %v\n", stateFile, err)
				} else {
					defer f.Close()
					for _, lf := range lms.lastFiltered {
						fmt.Fprintf(f, "%f\n", lf)
					}
				}
				return
			}
		}
	}()

	return nil
}

// filter processes the noisy signal, it is the consumer
func (fs *FilterSignal) filter(r *http.Request, plot *PlotT) error {
	// increment wg
	fs.wg.Add(1)

	// if no filter file specified, pass the samples unchanged
	filename := r.FormValue("filter")
	if len(filename) == 0 {
		fs.filterfile = "none"
		f2, _ := os.Create(path.Join(dataDir, signal))

		// launch a goroutine to no-filter generator signal
		// select on the generator semaphores and the done channel
		go func() {
			defer func() {
				f2.Close()
				fs.wg.Done()
			}()

			for {
				select {
				case n := <-fs.sema1:
					fs.nofilterBuf(0, n, f2)
				case n := <-fs.sema2:
					fs.nofilterBuf(1, n, f2)
				case <-fs.done:
					// if done signal from generator, save state, then return
					// save the Signal state:  time of last sample
					f, err := os.Create(path.Join(dataDir, stateFile))
					if err != nil {
						fmt.Printf("Create %s save state error: %v\n", stateFile, err)
					} else {
						defer f.Close()
						fmt.Fprintf(f, "%f\n", fs.lastSampleTime)
					}
					return
				}
			}
		}()
	} else {
		fs.filterfile = filename
		// get filter coefficients from file specified by user
		f, err := os.Open(path.Join(dataDir, filename))
		if err != nil {
			return fmt.Errorf("open %s error %v", filename, err)
		}
		defer f.Close()
		input := bufio.NewScanner(f)
		for input.Scan() {
			line := input.Text()
			h, err := strconv.ParseFloat(line, 64)
			if err != nil {
				return fmt.Errorf("filter coefficient conversion error: %v", err)
			}
			fs.filterCoeff = append(fs.filterCoeff, h)
		}

		// allocate memory for filtered output from previous submit
		fs.lastFiltered = make([]float64, len(fs.filterCoeff))

		f2, _ := os.Create(path.Join(dataDir, signal))

		// launch a goroutine to filter generator signal
		// select on the generator semaphores and the done channel
		go func() {
			defer f2.Close()
			defer fs.wg.Done()
			for {
				select {
				case n := <-fs.sema1:
					fs.filterBuf(0, n, f2)
				case n := <-fs.sema2:
					fs.filterBuf(1, n, f2)
				case <-fs.done:
					// if done signal from generator, save state, then return
					// save the Signal state:  time of last sample and incomplete filtered output
					f, err := os.Create(path.Join(dataDir, stateFile))
					if err != nil {
						fmt.Printf("Create %s save state error: %v\n", stateFile, err)
					} else {
						defer f.Close()
						fmt.Fprintf(f, "%f\n", fs.lastSampleTime)
						for _, lf := range fs.lastFiltered {
							fmt.Fprintf(f, "%f\n", lf)
						}
					}
					return
				}
			}
		}()
	}

	return nil
}

// showSineTable displays an empty sine table
func showSineTable(plot *PlotT) {
	// Fill in the table of frequencies and their amplitudes
	var (
		freqName []string = []string{"SSfreq1", "SSfreq2",
			"SSfreq3", "SSfreq4", "SSfreq5"}
		ampName []string = []string{"SSamp1", "SSamp2", "SSamp3",
			"SSamp4", "SSamp5"}
	)

	// show the sine table even if empty, otherwise it will never be filled in
	plot.Sines = make([]Attribute, len(freqName))
	for i := range freqName {
		plot.Sines[i] = Attribute{
			FreqName: freqName[i],
			AmplName: ampName[i],
			Freq:     "",
			Ampl:     "",
		}
	}
}

// label the plot and execute the PlotT on the HTML template
func (fs *FilterSignal) labelExec(w http.ResponseWriter, plot *PlotT) {

	plot.Xlabel = make([]string, xlabels)
	plot.Ylabel = make([]string, ylabels)

	// Construct x-axis labels
	incr := (fs.xmax - fs.xmin) / (xlabels - 1)
	x := fs.xmin
	// First label is empty for alignment purposes
	for i := range plot.Xlabel {
		plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (fs.ymax - fs.ymin) / (ylabels - 1)
	y := fs.ymin
	for i := range plot.Ylabel {
		plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}

	// Fill in the form fields
	plot.Samples = strconv.Itoa(fs.samples)
	plot.SampleFreq = strconv.Itoa(fs.sampleFreq)
	plot.SNR = strconv.Itoa(fs.snr)
	plot.Filename = path.Base(fs.filterfile)

	showSineTable(plot)

	i := 0
	//  Fill in any previous entries so the user doesn't have to re-enter them
	for _, sig := range fs.sines {
		plot.Sines[i].Freq = strconv.Itoa(sig.freq)
		plot.Sines[i].Ampl = strconv.Itoa(int(sig.ampl))
		i++
	}

	if len(plot.Status) == 0 {
		plot.Status = fmt.Sprintf("Signal consisting of %d sines was filtered with %s",
			len(fs.sines), plot.Filename)
	}

	// Write to HTTP using template and grid
	if err := filterSignalTmpl.Execute(w, plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}
}

// handleFilter filters a noisy signal using the ALE predictor
func handleFilterSignal(w http.ResponseWriter, r *http.Request) {
	var plot PlotT

	// need number of samples and sample frequency to continue
	temp := r.FormValue("samples")
	if len(temp) > 0 {
		samples, err := strconv.Atoi(temp)
		if err != nil {
			plot.Status = fmt.Sprintf("Samples conversion to int error: %v", err.Error())
			showSineTable(&plot)
			fmt.Printf("Samples conversion to int error: %v\n", err.Error())
			// Write to HTTP using template and grid
			if err := filterSignalTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		temp = r.FormValue("samplefreq")
		sf, err := strconv.Atoi(temp)
		if err != nil {
			fmt.Printf("Sample frequency conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Samples frequency conversion to int error: %v", err.Error())
			showSineTable(&plot)
			// Write to HTTP using template and grid
			if err := filterSignalTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Get FilterState from previous submit and store in FilterSignal
		var filterState FilterState
		// This file exists only after all the samples are generated and filtered
		f, err := os.Open(path.Join(dataDir, stateFile))
		if err == nil {
			defer f.Close()
			input := bufio.NewScanner(f)
			input.Scan()
			line := input.Text()
			// Last sample time from previous submit
			lst, err := strconv.ParseFloat(line, 64)
			if err != nil {
				fmt.Printf("From %s, sample time conversion error: %v\n", stateFile, err)
				plot.Status = fmt.Sprintf("Sample time conversion to int error: %v", err.Error())
				showSineTable(&plot)
				// Write to HTTP using template and grid
				if err := filterSignalTmpl.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			filterState.firstSampleTime = lst
			filterState.lastFiltered = make([]float64, 0)
			// Get the last incomplete filtered outputs from previous submit
			for input.Scan() {
				line := input.Text()
				fltOut, err := strconv.ParseFloat(line, 64)
				if err != nil {
					fmt.Printf("Sample time conversion error: %v\n", err)
					plot.Status = fmt.Sprintf("From %s, filtered output conversion to float error: %v", stateFile, err.Error())
					showSineTable(&plot)
					// Write to HTTP using template and grid
					if err := filterSignalTmpl.Execute(w, plot); err != nil {
						log.Fatalf("Write to HTTP output using template with error: %v\n", err)
					}
					return
				}
				filterState.lastFiltered = append(filterState.lastFiltered, fltOut)
			}
		} else {
			filterState = FilterState{firstSampleTime: 0.0, lastFiltered: make([]float64, 0)}
		}

		// create FilterSignal instance fs
		fs := FilterSignal{
			sema1:            make(chan int),
			sema2:            make(chan int),
			buf:              make([][]float64, 2),
			done:             make(chan struct{}),
			samples:          samples,
			samplesGenerated: 0,
			sampleFreq:       sf,
			sines:            make([]Sine, 0),
			FilterState:      filterState,
			filterCoeff:      make([]float64, 0),
		}
		fs.buf[0] = make([]float64, block)
		fs.buf[1] = make([]float64, block)

		// start generating samples and send to filter
		err = fs.generate(r)
		if err != nil {
			plot.Status = err.Error()
			showSineTable(&plot)
			fmt.Printf("generate error: %v\n", err.Error())
			// Write to HTTP using template and grid
			if err := filterSignalTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// start filtering when the samples arrive from the generator
		err = fs.filter(r, &plot)
		if err != nil {
			plot.Status = err.Error()
			showSineTable(&plot)
			fmt.Printf("filter error: %v\n", err.Error())
			// Write to HTTP using template and grid
			if err := filterSignalTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// wait for the generator and filter goroutines to complete
		fs.wg.Wait()

		if err != nil {
			plot.Status = err.Error()
		}

		// Fill in the PlotT grid with the signal time vs amplitude
		err = fs.gridFillInterp(&plot)
		if err != nil {
			plot.Status = err.Error()
			showSineTable(&plot)
			fmt.Printf("gridFillInterp error: %v\n", err.Error())
			// Write to HTTP using template and grid
			if err := filterSignalTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		//  generate x-labels, ylabels, status in PlotT and execute the data on the HTML template
		fs.labelExec(w, &plot)

	} else {
		// delete previous state file if initial connection (not a submit)
		if err := os.Remove(path.Join(dataDir, stateFile)); err != nil {
			// ignore error if file is not present
			fmt.Printf("Remove file %s error: %v\n", path.Join(dataDir, stateFile), err)
		}
		plot.Status = "Enter samples, sample frequency, SNR, frequencies and amplitudes"
		showSineTable(&plot)
		if err := filterSignalTmpl.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	}
}

// executive program
func main() {

	// Setup http server with handler for filtering a noisy signal using the ALE predictor
	http.HandleFunc(patternFilterSignal, handleFilterSignal)

	// Setup http server with handler for creating an Adaptive Line Enhancer predictor using LMS
	http.HandleFunc(patternLmsPredictorAle, handleLmsPredictorAle)

	// Setup http server with handler for plotting impulse or frequency responses for the ALE predictor
	http.HandleFunc(patternPlotResponse, handlePlotResponse)

	fmt.Printf("LMS ALE Predictor Server listening on %v.\n", addr)

	http.ListenAndServe(addr, nil)
}
