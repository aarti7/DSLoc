# Doppler Spread Localization Experiments



### Setup

We implement the following strategy for generating the fingerprinting database. One mobile transmitter



<!--### Measuring Doppler Spread using POWDER-->


<!--frequenices of operation-->
<!--calibration or synchorinization-->

### Analysis





<!--Locating signal source with signal received at multiple receiver nodes. (TDoA localization)

In this tutorial, we show how to perform a interference source localization experiment using the CBRS rooftop nodes. For this experiment, it is assumed that you have already created a POWDER account and have access to node reservations.
-->
<!--## Download files

We will need the following files to implement this tutorial, make sure to download them to the local computer:

- sp.py
- fabfile.py
<!--- hostfile.txt-->
<!--- interference_loc.ipynb

NOTE: A separate 'hostfile.txt' not needed, unlike for Fabric version 1. The 'fabfile.py' itself will have a dictionary `host_dict` to put the names (as keys) and hostname or IP address (as values) of the CBRS rooftop nodes. See 'Signal data collection'. --> 









<!--Once logged onto POWDER, navigate to:

**Experiments &rarr; Start Experiment &rarr; Change Profile &rarr; cir\_localization_update &rarr; Select Profile &rarr; Next**
-->
<!--Now we want to select the compute node type (d740) and base stations for the experiment. The frequency range you have access to is dependent on the project you belong to. For this experiment, a range of 10MHz between 3550MHz to 3700MHz is sufficient.

Any of the X310 CBRS Radio can be used. To add a radio, click the plus arrow and select a [location](https://powderwireless.net/area) from the drop down. A minimum of 3 radios is needed for TDoA localization(The more the better). Check [resource availibilty](https://www.powderwireless.net/resinfo.php?embedded=true) to find open radios or reserve them beforehand.

**&rarr; Next**

Now you can name your experiment (optional).

**&rarr; Next**

Create a start and end time for your experiment (optional). Leaving the fields empty keeps your experiment availible for 16 hours.

**&rarr; Finish**

## SSH into all the nodes

After your experiment is done setting up, SSH into the nodes.

## Upload the sp.py command script to all the nodes

From your local terminal use the following command:

(For Mac users) 

scp [file directory] [node link]:-->

<!--`scp /Users/sp.py joy@pc13-fort.emulab.net:`

(For Windows users) 

"C:\Program Files (x86)\PuTTY\pscp.exe" -scp [file directory] [node link]:

`"C:\Program Files (x86)\PuTTY\pscp.exe" -scp C:\Users/sp.py joy@pc13-fort.emulab.net:`

## Get access to the command script on all the nodes

On the nodes, use the following command:

`ls` (If the file was successfully uploaded, the filename will be listed out)

`sudo mv sp.py /usr/bin/` (Move the file to /usr/bin/)

`sudo chmod +x /usr/bin/sp.py` (Enable permission to execute file)

`sudo sysctl -w net.core.wmem_max=24862979` (Set time clocks synchronized)

## Signal data collection 

In order to collect data on all the nodes simultaneously, we use fabfile and hostsfile to control all the nodes from the local terminal.


You will require the `Fabric` package installed on your local machine.

It can be installed using the following command:
`pip install fabric` and it will install latest 'modern' Fabric, i.e. Fabric version 2.-->

<!--In hostsfile.txt, type in the nodes that you wish to receive the signal in the following form:

[name of the node] | RX | [link of the node]

honors | RX | ssh -p 22 joy@pc13-fort.emulab.net

hospital | RX | ssh -p 22 joy@pc15-fort.emulab.net

...and so on...-->

<!--In fabfile.py,


(assume password to be POWDER)


1. Change the ssh password to yours in the command line 

   `env.password = 'POWDER'`

2. In def grab_cir_measurements(rx_node_dict):, change the password to yours in the command line 
   
   `result = Connection(host,connect_kwargs={"password":"POWDER"}).get("disturb")`
1. Specify the avialable CBRS rooftop nodes in the `host_dict` by putting the names (as keys) and hostname or IP address (as values). Example:

	`host_dict = {'hospital': "joy@pc-780.emulab.net", 'smt': "joy@pc-772.emulab.net",  'ustar': "joy@pc-779.emulab.net"}`

3. Specify the parameters you want to use in RX FUNCTIONS.

   In 
   
    `def rx(rx_min):`
        
        `run("sp.py -f 3580e6 -s 200e6 -lo 100e6 -g 27 -d 0.1 -t %d" % rx_min)`

`-f`  : center frequency

`-s`  : sampling rate

`-lo` : lo offset (usually set to be slightly higher than half of sampling rate)

`-g`  : gain of the receiver

`-d`  : duration time of collection (in second)

* In the local terminal, use the following command: 
  `fab main`
* The collected signal files will be downloaded to the same directory as the location of `fabfile.py`.


<!--## Transfer measurement files from all the nodes to local
We will transfer the file named 'disturb' from each node in the experiment to the local machine and store each measurement by using the naming convention `rx_{name of the node}_raw`  
From your local terminal use the following command for each node:

`scp joy@pc13-fort.emulab.net:disturb ./rx_honors_raw` 
`scp joy@pc15-fort.emulab.net:disturb ./rx_hospital_raw` 

and so on..-->

<!--## Analysis

Now that you have finished reception, multiple complex binary files storing the signal IQ samples should appear in your directory. We can use the script `interference_loc.ipynb` to calculate and estimate the interference source location. The script calculate the time difference of arrival for each links and output a map of sum of squared errors between estimation and theoratical values. The location with the minimum sum of squared errors would be the source location with highest possibility. (Follow the mark downs to alter the set-ups for your case)
-->

