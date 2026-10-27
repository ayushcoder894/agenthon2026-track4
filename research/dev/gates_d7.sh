cd /work

export PYTHONPATH=/work/track4-analysis-public:/work/Agenthon2026-public/common
for u in $(ls track4-analysis-public/units); do
  echo "== $u"
  python -c "
import sys
from qfbench2_common.cli import main
sys.exit(main(['smoke', 'track4-analysis-public/units/$u', 't4-local/out/d7/$u', '--track', 'analysis']))" 2>&1 | tail -4 | cut -c1-230
done
