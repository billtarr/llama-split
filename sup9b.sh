#!/system/bin/sh
# supervisor: keep the 3-stage chain alive. exits the loop when /data/local/tmp/sup.stop exists
LOG=/data/local/tmp/sup.log
PORT=${1:-18080}
rm -f /data/local/tmp/sup.stop
while [ ! -f /data/local/tmp/sup.stop ]; do
    echo "[sup] start $(date) port=$PORT" >> $LOG
    sh /data/local/tmp/serve9b.sh $PORT
    rc=$?
    echo "[sup] chain exited rc=$rc at $(date)" >> $LOG
    [ -f /data/local/tmp/sup.stop ] && break
    sleep 3
done
echo "[sup] stopped $(date)" >> $LOG
