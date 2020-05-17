#!/bin/bash
# echo ${@} # Prints the IP addresses of the hosts.
count=0
for i in ${@//[[\],]}; do	
	count=$((count + 1))
	if (($count==1)); then
		continue
	fi
	until (nc -z ${i} 2345); do   
		sleep 1 # Waits for 1 second before checking again.
	done
done

count=0
for i in ${@//[[\],]}; do
	count=$((count + 1))
	if (($count==1)); then
		continue
	fi
	echo "hello" | netcat "${i}" 2345
done
echo "queue size:" > queue_size.txt

count=0
until (($count==50)); do
	#sleep 0.1 # Waits for 1 second before checking again.
	netcat -l 2345
	tc -s qdisc ls dev r-srv | grep -oP "Sent\s+\K\w+" >> queue_size.txt
	count=$((count + 1))
done
# tc -s qdisc ls dev r-eth3 | awk '{for(i=1;i<=NF;i++) if ($i=="Sent") print $(i+1)}' > queue_size.txt