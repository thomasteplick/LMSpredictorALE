# LMSpredictorALE
Least Mean Square (LMS) Adaptive Line Enhancer (ALE) predictor for filtering noisy, low SNR sinusoids.

This program is an HTTP web application written in Go that uses the html/template package to generate dynamic HTML.  Issue "go build predict.go" or issue "go run predict.go" to start the server. In a web browser enter http://127.0.0.1:8080/lmspredictorale in the address bar. This program will generate an adaptive FIR filter using the LMS algorithm.  It is used to filter a noisy signal with a low signal-to-noise ratio (SNR).  The signal is a sine wave in white noise with a supplied SNR.  Various parameters are specified.  The user can display the impulse and frequency response of the adaptive filter by clicking on the <i>Plot Response</i> link. The user can choose to display the filtered noisy sequence as well as the unfiltered sequence by clicking on the <i>Filter Signal</i> link.  Enter samples, sample rate, SNR, delay, trials, gain constant mu, iterations, and trials. The Least-Mean-Square algorithm will be used to create the adaptive filter. The filter weights will be averaged over trials, each trial consisting of iterations. The white Gaussian noise variance will be calculated from the SNR. The delay is the number of samples to delay the reference input to the adaptive filter. The sinsuoids, AM, or FM are the desired signals to be extracted from the noisy input. The adaptive filter is saved to disk and its impulse or frequency response can be displayed. Additionally, the weights can be used to filter another noisy signal. The links allow you to plot the impulse or frequency response and to use the adaptive filter to filter another noisy signal.

<h3>Select LMS Predictor Parameters</h3>

![image](https://github.com/thomasteplick/LMSpredictorALE/assets/117768679/7e56216c-3ce1-4aa3-898f-c4daf1e4ce85)

<h3>Plot the Impulse or Frequency Response of the Adaptive Filter</h3>

![image](https://github.com/thomasteplick/LMSpredictorALE/assets/117768679/eeb6f6d1-dc90-4786-acb3-abb17314965c)

<h3>Impulse Response of Adaptive Filter Zoom</h3>

![lmsPredictorALE](https://github.com/thomasteplick/LMSpredictorALE/assets/117768679/25d28746-42cf-454d-8f52-ab119ad5d915)

<h3>Impulse Response of Adaptive Filter</h3>

![lmsPredictorALE_2](https://github.com/thomasteplick/LMSpredictorALE/assets/117768679/9b10adcc-ff1a-4c82-aad9-46b144abc763)

<h3>Frequency Response of Adaptive Filter</h3>

![lmsPredictorALE_3](https://github.com/thomasteplick/LMSpredictorALE/assets/117768679/de5e0d70-7712-4e4a-8a87-f6aae54e14b5)


