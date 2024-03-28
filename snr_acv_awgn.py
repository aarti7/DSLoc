
def get_avg_power(samps):

    meanofthesignal = np.sum(samps) / len(samps)

    varofthesignal = np.sqrt( np.sum(np.square(samps - meanofthesignal)**2 ) / len(samps) ) 

    allpower = np.sum(np.square(np.abs(samps)))
    avgpwrlinear = allpower /len(samps) # expectecd value of S^2
    avgpwrindb = 10.0 * np.log10(avgpwrlinear)
    return avgpwrindb 


def get_acv1(corrected_signal_withfdANDfresidue, rate, nsamps, ns_freq_res):
    pdb.set_trace()

    meanofthesignal = np.sum(corrected_signal_withfdANDfresidue) / len(corrected_signal_withfdANDfresidue)

    varofthesignal = np.sqrt( np.sum(np.square(corrected_signal_withfdANDfresidue - meanofthesignal)**2 ) / len(corrected_signal_withfdANDfresidue) ) 

    sigma_squared_X = np.var(corrected_signal_withfdANDfresidue) # will be used to rep noise!!

    # Calculate the auto-correlation or auto-covariance of the signal+noise, and the contribution of the white noise is a scaled impulse,  𝜎2𝑋𝛿(𝑡) at the origin. 
    acv_frqcrctdlpfd = np.correlate(corrected_signal_withfdANDfresidue, corrected_signal_withfdANDfresidue, mode='full') / nsamps
    impulse = sigma_squared_X * np.array([1 if i == len(acv_frqcrctdlpfd) // 2 else 0 for i in range(len(acv_frqcrctdlpfd))])  # rep of  noise!!
    
    #The remaining auto-covariance is due to the signal. By removing the impulse and Fourier transforming the auto-covariance, you recover the spectrum of the “cleaned” signal.
    acv_awgn_rmvd_frqcrctdlpfd = acv_frqcrctdlpfd - impulse
    
    freqacv = np.fft.fftshift(np.fft.fftfreq(len(awgn_rmvd_frqcrctdlpfd), 1/rate))
    fdidxacv  = (freqacv >= -ns_freq_res) & ( freqacv <= ns_freq_res)


    return acv_frqcrctdlpfd, acv_awgn_rmvd_frqcrctdlpfd, fdidxacv

def get_acv2(corrected_signal_withfdANDfresidue, rate, nsamps, ns_freq_res):
    pdb.set_trace()

    meanofthesignal = np.sum(corrected_signal_withfdANDfresidue) / len(corrected_signal_withfdANDfresidue)

    varofthesignal = np.sqrt( np.sum(np.square(corrected_signal_withfdANDfresidue - meanofthesignal)**2 ) / len(corrected_signal_withfdANDfresidue) ) 

    sigma_squared_X = np.var(corrected_signal_withfdANDfresidue) # will be used to rep noise!!

    # Calculate the auto-correlation or auto-covariance of the signal+noise, and the contribution of the white noise is a scaled impulse,  𝜎2𝑋𝛿(𝑡) at the origin. 
    acv_frqcrctdlpfd = np.correlate(corrected_signal_withfdANDfresidue, corrected_signal_withfdANDfresidue, mode='full') / nsamps
    impulse = sigma_squared_X * np.array([1 if i == len(acv_frqcrctdlpfd) // 2 else 0 for i in range(len(acv_frqcrctdlpfd))])  # rep of  noise!!
    
    #The remaining auto-covariance is due to the signal. By removing the impulse and Fourier transforming the auto-covariance, you recover the spectrum of the “cleaned” signal.
    acv_awgn_rmvd_frqcrctdlpfd = acv_frqcrctdlpfd - impulse
    
    freqacv = np.fft.fftshift(np.fft.fftfreq(len(awgn_rmvd_frqcrctdlpfd), 1/rate))
    fdidxacv  = (freqacv >= -ns_freq_res) & ( freqacv <= ns_freq_res)


    return acv_frqcrctdlpfd, acv_awgn_rmvd_frqcrctdlpfd, fdidxacv

def get_acv3(sig, fs):
    pdb.set_trace()
   
    """    
    # mean
    meanofthesignal = np.sum(sig) / len(sig)
    
    #var
    v0 = np.sum(np.square(sig - meanofthesignal) )
    v1 = v0 / len(sig)
    varofthesignal  = np.linalg.norm( v1 )

    # var auto
    sigma_squared_X = np.var(sig) # will be used to rep noise!!

    # check if same
    sigma_squared_X == varofthesignal

    allpower = np.sum(np.square(np.abs(samps))) #  were samps are already mean rmeoved?N0 i think cause mean was calculated above and it was nonzero. abs(x_i) = [sqrt(a^2 + b^2)_i]
    avgpwrlinear = allpower /len(samps) # expectecd value of S^2
    # Ans: so for power and var to be same, remove mean from sia before taking the power!
    """
    avg = np.mean(sig)
    var = np.var(sig) # doesn't change when mean is removed from sig

    sig = sig - avg
    allpower = np.sum(np.square(np.abs(sig))) 
    avgpwrlinear = allpower /len(sig) # must be equal to the var
    avgpwrindb = 10.0 * np.log10(avgpwrlinear)

    """
    1. fs samples take   --> 1 sec 
       one sample takes  --> 1/fs seconds --> ts seconds
       so:     t ->>> np.arange(0, 124*ts, ts) ---- > will give a t of lengthof_t=124== durationofsig, that is, 120 smaples with space of ts between them.

       you know fs == 50 samples per second here and you know 124 as such before hand. 

    2.  how to get this same t vector by linspace?

        t --->  np.linspace(0, 1sec, fs) -- > makes sense by defination of frequency being numnr of samples in one second 
        you can think of endtimeofsig = 1second here

        more imrpotantly, this will create a t vecot of length=1sec*fs
        t ---> np.linspace(0, endtime=1sec, 50,endpoint=False) same length of t==1second*50==== >50

        if endtimeofsig = 2.48seconds
        Numberfsamplesbythisendtime = fs * endtimeofsig == length of t vector ==> thus, thrid place shuld be equal to 2.48*fs= 124
        
        t --> np.linspace(0,  endtime= 2.48inseconds, 2.48*fs= 124 samplesbytheendtime) =len(t)is 124

        so more genreally, t = np.linspace(0,  E*1, E*fs) such that lengthof_t ==124

        summary:
        np.arange(0, L*ts, ts) ==t== np.linspace(0,  1*E, , int(E*fs), endpoint=False)
        len(t) = L = 124
        so: E = L*ts = 2.48

        # Calculate the auto-correlation or auto-covariance of the signal+noise, and the contribution of the white noise is a scaled impulse,  𝜎2𝑋𝛿(𝑡) at the origin. 
        sigma_squared_X = np.var(sig)

        acv_frqcrctdlpfd = np.correlate(sig, sig, mode='full') / len(sig)
        impulse = sigma_squared_X * np.array([1 if i == len(acv_frqcrctdlpfd) // 2 else 0 for i in range(len(acv_frqcrctdlpfd))])  # rep of  noise!!
    """




    
    #The remaining auto-covariance is due to the signal. By removing the impulse and Fourier transforming the auto-covariance, you recover the spectrum of the “cleaned” signal.
    clean_rxx = acv_frqcrctdlpfd - impulse
    clean_time = np.fft.ifft(np.fft.fftshift(clean_rxx)) 
    freqacv = np.fft.fftshift(np.fft.fftfreq(len(acv_awgn_rmvd_frqcrctdlpfd), 1/fs))
    fdidxacv  = (freqacv >= -ns_freq_res) & ( freqacv <= ns_freq_res)

    return acv_frqcrctdlpfd, acv_awgn_rmvd_frqcrctdlpfd, fdidxacv

 
def get_SNR():
    plot the SNR of each observation over the 2d xy for each basestation in the 3d plot.
    I think the sNR should show a pattern according to the distance of the observation from the BS


def plot_sig(this_measurement, this_measurement_after_LPF, corrected_signal_withfdANDfresidue, fdidx, acv_frqcrctdlpfd, awgn_rmvd_frqcrctdlpfd, fdidxacv, rate, nsamps):

    # np.random.seed(42)
    # t = np.linspace(0, 1, len(this_measurement), endpoint=False)
    # clean_signal = np.max(this_measurement.real)*np.sin(2 * np.pi * 50 * t)+np.max(this_measurement.imag)* 1j*np.sin(2 * np.pi * 50 * t)

    # # Adaptive Filtering (LMS algorithm)
    # mu = 0.01  # Adaptation step size
    # order = 32  # Filter order
    # # Initialize filter weights
    # W = np.zeros(order)
    # pdb.set_trace()

    # # LMS algorithm
    # for i in range(order, len(this_measurement)):
    #     x = this_measurement[i-order:i]
    #     y_hat = np.dot(W, x)
    #     e = clean_signal[i] - y_hat
    #     W = W + mu * e * x

    # clean_estimate = np.convolve(this_measurement, W, mode='same')

    # plt.figure(figsize=(10, 6))
    # plt.plot(t, clean_signal, label='Clean Signal', alpha=0.7)
    # plt.plot(t, this_measurement, label='Noisy Signal', alpha=0.7)
    # plt.plot(t, clean_estimate, label='Clean Estimate', linestyle='--', linewidth=2)
    # plt.legend()
    # plt.title('Adaptive Filtering for AWGN Denoising (LMS)')
    # plt.show()

    fig, ax = plt.subplots(3,4)
    ax[0][0].plot(this_measurement, label = "raw")
    ax[0][1].plot(this_measurement_after_LPF, label = "lpfd")
    ax[0][2].plot(corrected_signal_withfdANDfresidue, label="frqcrctd+lpfd")
    ax[0][3].plot(np.arange(-len(acv_frqcrctdlpfd)//2, len(acv_frqcrctdlpfd)//2), acv_frqcrctdlpfd, label = "frqcrctd+lpfd acv")

    ax[1][0].plot(np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate)), np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(this_measurement)))))), label = "raw fft")
    ax[1][1].plot(np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate)), np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(this_measurement_after_LPF)))))), label = "lpfd fft")
    ax[1][2].plot(np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate)), np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(corrected_signal_withfdANDfresidue)))))), label="frqcrctd+lpfd fft")
    ax[1][3].plot(np.fft.fftshift(np.fft.fftfreq(len(awgn_rmvd_frqcrctdlpfd), 1/rate)), np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(awgn_rmvd_frqcrctdlpfd)))))), label = "frqcrctd+lpfd awgnremoved fft full")
    
    ax[2][0].plot(np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate))[fdidx], np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(this_measurement))))))[fdidx], label = "raw fft ns")
    ax[2][1].plot(np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate))[fdidx], np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(this_measurement_after_LPF))))))[fdidx], label = "lpfd fft ns")
    ax[2][2].plot(np.fft.fftshift(np.fft.fftfreq(nsamps, 1/rate))[fdidx], np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(corrected_signal_withfdANDfresidue))))))[fdidx], label="frqcrctd+lpfd fft ns")
    ax[2][3].plot(np.fft.fftshift(np.fft.fftfreq(len(awgn_rmvd_frqcrctdlpfd), 1/rate))[fdidxacv], np.nan_to_num(10.0*np.log10(np.square(np.abs(np.fft.fftshift(np.fft.fft(awgn_rmvd_frqcrctdlpfd))))))[fdidxacv], label="frqcrctd+lpfd  awgn removed fft ns")


    ax[0][0].legend(loc="lower left"), ax[0][1].legend(loc="lower left"), ax[0][2].legend(loc="lower left"), ax[1][0].legend(loc="lower left")
    ax[1][1].legend(loc="lower left"), ax[1][2].legend(loc="lower left"), ax[2][0].legend(loc="lower left"), ax[2][1].legend(loc="lower left"),
    ax[2][2].legend(loc="lower left"), ax[0][3].legend(loc="lower left"), ax[1][3].legend(loc="lower left"), ax[2][3].legend(loc="lower left"), 
    ax[1][0].set_ylim(-180,20),ax[1][1].set_ylim(-180,20),ax[1][2].set_ylim(-180,20),ax[2][0].set_ylim(-80,20),ax[2][1].set_ylim(-80,20),ax[2][2].set_ylim(-80,20) # ax[3][1].set_ylim(-80,20), ax[3][2].set_ylim(-80,20)
    plt.legend()
    plt.show()




