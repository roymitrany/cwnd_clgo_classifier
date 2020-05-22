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
echo "queue size:" > /tmp/qqq.txt

count=0
until (($count==500)); do
	sleep 0.1 # Waits for 1 second before checking again.
	#netcat -l 2345
	#tc -s qdisc show dev r-srv| grep backlog  >> /tmp/qqq.txt
	#python queue_len_poller.py r-srv >> /tmp/qqq.txt
	count=$((count + 1))
done
# tc -s qdisc ls dev r-eth3 | awk '{for(i=1;i<=NF;i++) if ($i=="Sent") print $(i+1)}' > queue_size.txt