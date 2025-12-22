import os
import pandas as pd
from utils_excel import append_to_excel

TEST_FILE = 'test_metrics.xlsx'

def test_append():
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)
        
    # Test 1: Create new file
    data1 = {
        'Model': 'TestModel',
        'Accuracy': 99.9
    }
    print("Appending first row...")
    append_to_excel(TEST_FILE, data1)
    
    assert os.path.exists(TEST_FILE)
    df = pd.read_excel(TEST_FILE)
    assert len(df) == 1
    assert df.iloc[0]['Model'] == 'TestModel'
    
    # Test 2: Append to existing
    data2 = {
        'Model': 'TestModel2',
        'Accuracy': 88.8
    }
    print("Appending second row...")
    append_to_excel(TEST_FILE, data2)
    
    df = pd.read_excel(TEST_FILE)
    assert len(df) == 2
    assert df.iloc[1]['Accuracy'] == 88.8
    
    print("Verification successful!")
    # Cleanup
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)

if __name__ == "__main__":
    test_append()
