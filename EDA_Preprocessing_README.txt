EDA & PREPROCESSING NOTES
   -------------------------
   Date: 2025-10-29 16:04

   Target:
     - default = 1: Charged Off, Default, Late (31-120 days), Late (16-30 days),
       Does not meet the credit policy. Status:Charged Off
     - default = 0: Fully Paid, Does not meet the credit policy. Status:Fully Paid
     - Others dropped.

   Selected features:
      - loan_amnt: Requested principal
- funded_amnt: Actually funded principal
- term: Loan term (36/60 months)
- installment: Monthly EMI
- int_rate: Interest rate (%)
- annual_inc: Annual income
- dti: Debt-to-Income
- emp_length: Employment tenure
- home_ownership: Home ownership
- verification_status: Income/Employment verified?
- purpose: Loan purpose
- addr_state: Borrower state
- revol_bal: Revolving balance
- revol_util: Revolving usage (%)
- open_acc: Open accounts
- total_acc: Total accounts
- delinq_2yrs: Delinquencies (2 yrs)
- inq_last_6mths: Credit inquiries (6 mo)
- pub_rec: Derogatory public records

   Cleaning:
     - Converted percent strings (int_rate, revol_util) to float
     - Parsed emp_length to numeric years
     - Parsed dates (issue_d, earliest_cr_line)
     - Engineered: issue_year, issue_month, issue_ym, credit_hist_months
     - Dropped raw date columns post-engineering

   Split:
     - Time-aware 80/20 by issue_d (older -> train, newer -> test)

   Preprocessing:
     - Numeric: median impute + StandardScaler
     - Categorical: most_frequent impute + OneHotEncoder(handle_unknown='ignore')

   Files created:
     - loan_clean_subset.csv         (engineered features + target)
     - loan_clean_sample_50k.csv     (random sample up to 50k rows)
     - EDA_Preprocessing_README.txt  (this file)