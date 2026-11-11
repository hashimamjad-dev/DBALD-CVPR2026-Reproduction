import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_isolated_cpu_checks import run_child, classify


def test_monitor_counts_the_real_interpreter_and_stops_only_its_own_child(tmp_path):
    command=[sys.executable,'-c','import time; data=bytearray(25*1024**2); time.sleep(.5)']
    result=run_child(command,tmp_path/'pass.log',os.environ.copy(),256,10)
    assert result['status']=='PASSED' and result['peak_rss_mib']>20
    result=run_child([sys.executable,'-c','import time; time.sleep(5)'],tmp_path/'timeout.log',os.environ.copy(),256,.2)
    assert result['status']=='TIMEOUT'
    assert classify(3221225477)=='NATIVE_ACCESS_VIOLATION' and classify(3)=='FAILED'
