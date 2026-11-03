from src.generate_data import generate_synthetic_data
from src.analysis import run_analysis
if __name__ == "__main__":
    p,c,e=generate_synthetic_data()
    p.to_csv("data/raw/patients.csv",index=False)
    c.to_csv("data/raw/pharmacy_claims.csv",index=False,date_format="%Y-%m-%d")
    e.to_csv("data/raw/enrollment.csv",index=False,date_format="%Y-%m-%d")
    print(f"Generated {len(p):,} patients, {len(c):,} claims, {len(e):,} enrollment spans.")
    s,ss=run_analysis()
    print("\nSUMMARY")
    for k,v in s.items(): print(f"{k}: {v}")
    print("\nSENSITIVITY\n",ss.to_string(index=False))
