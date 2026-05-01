import pandas as pd

def compare_results():
    sample_df = pd.read_csv('../support_tickets/sample_support_tickets.csv')
    output_df = pd.read_csv('../support_tickets/output.csv')
    
    print(f"{'Row':<4} | {'Exp Status':<12} | {'Act Status':<12} | {'Exp Type':<15} | {'Act Type':<15}")
    print("-" * 75)
    
    matches_status = 0
    matches_type = 0
    
    for i in range(len(sample_df)):
        expected_status = str(sample_df.iloc[i]['Status']).lower()
        actual_status = str(output_df.iloc[i]['status']).lower()
        
        expected_type = str(sample_df.iloc[i]['Request Type']).lower()
        actual_type = str(output_df.iloc[i]['request_type']).lower()
        
        status_marker = "[OK]" if expected_status == actual_status else "[XX]"
        type_marker = "[OK]" if expected_type == actual_type else "[XX]"
        
        if expected_status == actual_status: matches_status += 1
        if expected_type == actual_type: matches_type += 1
        
        print(f"{i+1:<4} | {expected_status:<12} | {actual_status:<12} {status_marker} | {expected_type:<15} | {actual_type:<15} {type_marker}")
    
    print("-" * 75)
    print(f"Status Accuracy: {matches_status}/{len(sample_df)} ({matches_status/len(sample_df)*100:.1f}%)")
    print(f"Request Type Accuracy: {matches_type}/{len(sample_df)} ({matches_type/len(sample_df)*100:.1f}%)")

if __name__ == '__main__':
    compare_results()
