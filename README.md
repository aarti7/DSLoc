# Doppler spread based localization
We implement the following strategy for generating the fingerprint database $F$. 

We perform a series of large-scale, over-the- air experiments on the POWDER platform \cite{}. In total $N=5$ base-stations ( namely Behavioral, Honors, Hospital, SMT, UStar) are utilized.  

### Experiments

A mobile node continuously transmits an unmodulated, complex sinusoidal wave at a frequency $f_c = 3515$ MHz. The bandwidth of operation is $10$ MHz. The B210 radios that are used for transmission, itself do not have a fixed power level of transmission, as the power level is determined by the RF front-end and the antenna used with the SDR. The AD9361 RFIC transceiver used in the B210 provides an adjustable and programmable output power ranging from -4 dBm to 6 dBm.

 Multiple base-stations simultaneously record the IQ values during the duration of each of the experiments. Each base-station has a USRP X310 receiver. The power of operation is and the receive gain is maintained as 30 dB. All the receiver X310 have an external clock of base-stations are time and frequency synchronized with the White Rabbit Synchronization system \cite{}.




<!--Measuring Doppler Spread using POWDER-->
<!--frequenices of operation-->
<!--calibration or synchorinization-->




SHOUT: 
piping issues: measurements dropped.
More robust signal reception.


### Data analysis


