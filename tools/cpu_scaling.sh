# scale to high performance
sudo pstate-frequency -S -g performance --min 99

# scale to powersave performance
sudo pstate-frequency -S -g powersave --min 30

# watch actual frequency
 watch -n.1 "cat /proc/cpuinfo | grep \"^[c]pu MHz\""