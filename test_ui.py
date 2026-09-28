"""Optional integration test; run with python test_ui.py after installing requirements."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

if __name__=='__main__':
    app=AppTest.from_file(str(Path(__file__).with_name('app.py')),default_timeout=60).run()
    assert not app.exception, app.exception
    app.sidebar.checkbox[0].check().run()
    assert not app.exception, app.exception
    assert len(app.metric)==3
    assert len(app.dataframe)==4
    app.sidebar.selectbox[1].select('time').run()
    assert not app.exception
    assert len(app.error)==1
    print('PASS: empty state, demo, three forecasts, row/event evaluation, export, invalid column handling')
