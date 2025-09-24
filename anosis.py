# ─────────── 자동 배치 스크립트 (save_as whatever.py) ───────────
import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import norm
import numpy as np
import shutil
# C://Users//MCNEX//AppData//Roaming//Microsoft//Excel//hong//anosis
# ① Raw714 위치
# BASE_DIR = r"C://Users//MCNEX//AppData//Roaming//Microsoft//Excel//hong//kakao_save_file//Data_A26_Daily//Data_A26_Daily//Raw714"
BASE_DIR = r"C://Users//MCNEX//AppData//Roaming//Microsoft//Excel//hong//anoysis_9_22_NIGHT"


# ② 분류별(폴더) 설정 : 열 인덱스 → 새 이름 , 사양(LSL,USL)
##ㄴ데이터 매핑

CATEGORY_CFG = {
    "AF": {
        "rename": {21: "0_Hysteresis", 37: "180_Hysteresis", 22: "0_linearity", 38: "180_linearity"},
        "spec"  : {"0_Hysteresis": (-6, 6), "180_Hysteresis": (-6, 6),
                   "0_linearity": (-10,10), "180_linearity": (-10,10)}, },
    
    "TILT": {
        "rename": {7: "MAX_TILT_0", 8: "MAX_TILT_180"},
        "spec"  : {"MAX_TILT_0": (0, 10), "MAX_TILT_180": (0,13)}, }
    
    
    
    
}

# ───────── 함수들 ─────────
def pretty(x,p=2): return f"{x:.{p}f}" if x is not None else "-"

def analyze_series(data, LSL, USL):
    
    
    # s = pd.to_numeric(s, errors='coerce').dropna()
    
    data = pd.to_numeric(data, errors="coerce").dropna()
    # data = pd.to_numeric(data, errors="coerce")
    
    # series = series.fillna(0)
    # if s.empty: return None
    # mu, sd, n = s.mean(), s.std(), len(s)
    # cpl = (mu-LSL)/(3*sd) if LSL is not None else None
    # cpu = (USL-mu)/(3*sd) if USL is not None else None
    # cpk = min(filter(None,[cpl,cpu])) if any([cpl,cpu]) else None
    # return dict(N=n, Min=s.min(), Max=s.max(), Avg=mu, Cpk=cpk, SD=sd)


    # mean_val = data.mean()
    mean_val = data.abs().mean()
    std_sample = data.std()
    std_pop = data.std(ddof=0)
    n = len(data)
    min_data = min(data)
    max_data = max(data)

    cpl = (mean_val - LSL) / (3 * std_sample) if LSL is not None else None
    cpu = (USL - mean_val) / (3 * std_sample) if USL is not None else None
    
    cpk = min(cpl, cpu) if cpl is not None and cpu is not None else cpl or cpu
    
    return dict(N=n, Min=min_data, Max=max_data, Avg=mean_val, Cpk=cpk, SD=std_sample)
    # return dict(N=n, Min=min_data, Max=max_data, Avg=mean_val, Cpk=cpk, SD=std_pop)

def plot_save(series, title, LSL, USL, avg, cpk, min_val, max_val, out_png):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10,6))
    
    # 히스토그램
    sns.histplot(series, bins=19, stat="density",
                 color="lightgray", edgecolor="black")
    
    # mu, std = np.mean(series), np.std(series, ddof=1)
    mu, std = np.mean(series), np.std(series, ddof=0)
    
    
    

    if std > 0:  # 표준편차가 0이면 pdf 그릴 수 없음
        # 기존: x = np.linspace(min(series), max(series), 200)
        # 수정: 평균 ± 4σ 범위로 넓게 설정
        x = np.linspace(mu - 4*std, mu + 4*std, 400)
        y = norm.pdf(x, mu, std)
        plt.plot(x, y, 'k-', linewidth=2.5)
        plt.plot(x, y, 'r--', linewidth=2.0)
        
    # LSL, USL
    if LSL is not None: 
        plt.axvline(LSL, color='red', ls='--', label=f"LSL={pretty(LSL)}")
    if USL is not None: 
        plt.axvline(USL, color='red', ls='--', label=f"USL={pretty(USL)}")
    
    # 평균
    if avg is not None: 
        plt.axvline(avg, color='blue', ls='--', label=f"Avg={pretty(avg,2)}")
    

    
    # Cpk, Min, Max (텍스트만 범례에 표시)
    if cpk is not None: 
        plt.axhline(y=0, color='white', alpha=0, label=f"Cpk={pretty(cpk,3)}")
        
    if min_val is not None: 
        plt.axhline(y=0, color='black', alpha=0, label=f"min={pretty(min_val,2)}")
        
    if max_val is not None: 
        plt.axhline(y=0, color='black', alpha=0, label=f"max={pretty(max_val,2)}")
    
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()



def deduplicate_columns(columns):
    seen = {}
    new_cols = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            new_cols.append(col)
        else:
            new_col = f"{col}.{seen[col]}"
            seen[col] += 1
            new_cols.append(new_col)
    return new_cols



# ───────── 메인 루프 ─────────
summary = []

result_dir = os.path.join(BASE_DIR, "ResultPlot")
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir, exist_ok=True)



for cat, cfg in CATEGORY_CFG.items():
    cat_dir = os.path.join(BASE_DIR, cat)
    if not os.path.isdir(cat_dir):
        print(f"‣ 폴더 없음 → {cat_dir}"); continue
    for fname in os.listdir(cat_dir):
        if not fname.lower().endswith((".xls",".xlsx")): continue
        fpath = os.path.join(cat_dir,fname)
        try:
            df = pd.read_excel(fpath, header=0)         # 헤더행(4번째) 맞는지 확인!
            
            
            
            
            
            
            df.columns = deduplicate_columns(df.columns)
            
            # df = df.rename(columns={df.columns[i]:new for i,new in cfg["rename"].items()})
            
            rename_map = {}

            if cat == "FRA":
                df.columns = deduplicate_columns(df.columns)

                rename_map = {}

                # OverGain 처리
                if "X_OverGain.1" in df.columns:
                    rename_map["X_OverGain.1"] = "X_OverGain"
                if "X_OverGain" in df.columns:
                    df = df.drop(columns=["X_OverGain"])

                if "Y_OverGain.1" in df.columns:
                    rename_map["Y_OverGain.1"] = "Y_OverGain"
                if "Y_OverGain" in df.columns:
                    df = df.drop(columns=["Y_OverGain"])

                # PlantPeakFreq+ → PlantPeakFreq
                if "Y_PlantPeakFreq+" in df.columns:
                    rename_map["Y_PlantPeakFreq+"] = "Y_PlantPeakFreq"
                if "X_PlantPeakFreq+" in df.columns:   # 혹시라도 있을 경우
                    rename_map["X_PlantPeakFreq+"] = "X_PlantPeakFreq"

                df = df.rename(columns=rename_map)
                
                
            else:
                # AF, OIS는 기존 방식
                rename_map = {}
                for idx, new_name in cfg["rename"].items():
                    if idx < len(df.columns):
                        rename_map[df.columns[idx]] = new_name
                df = df.rename(columns=rename_map)

    
    
            


            #######검사오류샘플 데이터 일괄 제외 코드
            ####### 검사오류샘플 데이터 일괄 제외 + Rawdata 사전 필터링
            sub = df[list(cfg["spec"].keys())].dropna()

            filtered = pd.DataFrame()
            for col, (lsl, usl) in cfg["spec"].items():
                if col not in sub.columns:
                    continue
                series = pd.to_numeric(sub[col], errors="coerce").dropna()

                # ── Rawdata 전처리 필터링 조건 적용 ──
                if col in ["0_Hysteresis", "180_Hysteresis"]:
                    series = series[(series >= -4.2) & (series <= 7.8)]
                elif col in ["0_linearity", "180_linearity"]:
                    series = series[(series >= -7) & (series <= 13)]
                    
                elif col in ["MAX_TILT_0"]:
                    series = series[(series >= 7) & (series <= 13)]

                elif col in ["MAX_TILT_180"]:
                    series = series[(series >= 9.1) & (series <= 16.9)]








                if not series.empty:
                    filtered[col] = series

            # ── Spec (LSL/USL) 기준으로 analyze_series 실행 ──
            for col,(lsl,usl) in cfg["spec"].items():
                if col not in filtered.columns:
                    continue
                res = analyze_series(filtered[col], lsl, usl)
                if res is None: 
                    continue
                res.update(Category=cat, File=os.path.splitext(fname)[0], Metric=col)
                summary.append(res)
                png = os.path.join(BASE_DIR,"ResultPlot",cat,res["File"],f"{col}.png")
                plot_save(filtered[col], f"{col} ({res['File']})", lsl, usl,
                          res["Avg"], res["Cpk"], res["Min"], res["Max"], png)
                
        except Exception as e:
            print(f"⚠ 파일 처리 실패 {fpath} → {e}")




# ───────── 요약 엑셀 저장 ─────────
if summary:
    summary_df = pd.DataFrame(summary)[["Category","File","Metric","N","Min","Max","Avg","Cpk"]]
    out_excel = os.path.join(BASE_DIR,"ResultPlot","Summary_Report.xlsx")
    os.makedirs(os.path.dirname(out_excel), exist_ok=True)
    summary_df.to_excel(out_excel, index=False)
    print("✅ 요약 저장 →", out_excel)
else:
    print("⚠ 분석된 데이터가 없습니다. 헤더 위치·열 인덱스 확인하세요.")
# ───────────────────────────────────────────────────────────
