# 📥 Submissions: TCGA-LGG Challenge

This directory holds official encrypted participant submissions.

## Rules for Submitting:
1. **Name your file with your team or participant name**: `<team_name>.csv.enc` (e.g. `emmanuel_owusu.csv.enc`).
2. **Never commit the unencrypted `.csv` file**: The GitHub auto-grader will reject unencrypted files.
3. **Format**: Your decrypted CSV must adhere to:
   ```csv
   patient_id,slice_idx,rle_mask
   ```
4. **How to submit**:
   ```bash
   # 1. Encrypt your CSV
   python starter_code/encrypt.py submissions/<team_name>.csv

   # 2. Add and commit only the .enc file
   git add submissions/<team_name>.csv.enc
   git commit -m "Submit encrypted predictions for <team_name>"
   git push origin <your-branch>
   ```
5. Open a **Pull Request** against `main`. The Auto-Grader bot will decrypt, validate, grade, and comment your official score on the PR!
