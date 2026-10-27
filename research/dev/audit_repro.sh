cd /d/AGENTHON
for u in $(ls track4-analysis-public/units | grep -v -E "EXAMPLE|units\$"); do
  out=t4-local/exp_out/d20_audit/repro/$u; rm -rf $out; mkdir -p $out
  s=$(date +%s)
  MSYS_NO_PATHCONV=1 timeout 700 docker run --rm --gpus all --network none -v "D:/AGENTHON/track4-analysis-public/units/$u:/input:ro" -v "D:/AGENTHON/$out:/output" agenthon-t4-agent:d20r-local analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json > $out.log 2>&1
  echo "$u exit $? $(( $(date +%s) - s ))s" >> t4-local/exp_out/d20_audit/repro_status.log
done
echo REPRO_DONE >> t4-local/exp_out/d20_audit/repro_status.log
