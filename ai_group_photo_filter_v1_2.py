# -*- coding: utf-8 -*-
import os,sys,csv,shutil,threading,queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import cv2,numpy as np
from PIL import Image,ImageOps
import pillow_heif
pillow_heif.register_heif_opener()
MODEL="face_detection_yunet_2023mar.onnx"
EXTS={".jpg",".jpeg",".png",".bmp",".webp",".heic",".heif",".tif",".tiff"}
CATS=[("00_未偵測到人臉",lambda n:n==0),("01_單人照",lambda n:n==1),("02_雙人照",lambda n:n==2),("03_小組照_3-5人",lambda n:3<=n<=5),("04_團體照_6-10人",lambda n:6<=n<=10),("05_大合照_10人以上",lambda n:n>=11)]
def model_path():
    return Path(getattr(sys,"_MEIPASS",Path(__file__).parent))/MODEL
def image(p):
    with Image.open(p) as im:
        im=ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size)>3200:
            s=3200/max(im.size);im=im.resize((int(im.width*s),int(im.height*s)))
        return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
def unique(d,p):
    x=d/p.name;i=2
    while x.exists():x=d/f"{p.stem}_{i}{p.suffix}";i+=1
    return x
class App(tk.Tk):
    def __init__(self):
        super().__init__();self.title("AI合照快速篩選器 v1.2｜YuNet 合照快速模式");self.geometry("900x650")
        self.src=tk.StringVar();self.dst=tk.StringVar();self.fm=tk.StringVar(value="只找合照：3人以上")
        self.sens=tk.StringVar(value="團體照優先");self.rec=tk.BooleanVar(value=True);self.q=queue.Queue();self.stop=False;self.p=tk.DoubleVar()
        ttk.Label(self,text="AI 合照快速篩選器 v1.2",font=("Microsoft JhengHei UI",20,"bold")).pack(pady=(18,2))
        ttk.Label(self,text="YuNet 本機 AI｜新增 3+／6+／10+ 一鍵合照篩選｜照片不上傳").pack()
        f=ttk.Frame(self);f.pack(fill="x",padx=18,pady=12)
        for r,(t,v,c) in enumerate([("來源照片：",self.src,self.picksrc),("輸出位置：",self.dst,self.pickdst)]):
            ttk.Label(f,text=t).grid(row=r,column=0,pady=5);ttk.Entry(f,textvariable=v).grid(row=r,column=1,sticky="ew",padx=8);ttk.Button(f,text="選擇資料夾",command=c).grid(row=r,column=2)
        f.columnconfigure(1,weight=1)
        o=ttk.LabelFrame(self,text="快速篩選設定");o.pack(fill="x",padx=18,pady=5)
        ttk.Label(o,text="輸出模式：").grid(row=0,column=0,padx=8,pady=10)
        ttk.Combobox(o,textvariable=self.fm,state="readonly",width=30,values=["完整分類","只找合照：3人以上","只找團體照：6人以上","只找大合照：10人以上"]).grid(row=0,column=1)
        ttk.Label(o,text="YuNet 靈敏度：").grid(row=0,column=2,padx=(25,8))
        ttk.Combobox(o,textvariable=self.sens,state="readonly",width=18,values=["精準優先","平衡","團體照優先"]).grid(row=0,column=3)
        ttk.Checkbutton(o,text="包含子資料夾",variable=self.rec).grid(row=1,column=0,columnspan=2,padx=8,pady=(0,8),sticky="w")
        ttk.Label(o,text="安全設定：只複製符合條件的照片，不移動、不刪除原始照片。").grid(row=1,column=2,columnspan=2,padx=8,sticky="w")
        ttk.Progressbar(self,variable=self.p,maximum=100).pack(fill="x",padx=18,pady=(15,4))
        self.status=ttk.Label(self,text="請選擇照片資料夾");self.status.pack(anchor="w",padx=18)
        self.log=tk.Text(self,height=18);self.log.pack(fill="both",expand=True,padx=18,pady=8)
        b=ttk.Frame(self);b.pack(fill="x",padx=18,pady=10)
        self.go=ttk.Button(b,text="開始 YuNet AI 篩選",command=self.start);self.go.pack(side="left")
        ttk.Button(b,text="停止",command=lambda:setattr(self,"stop",True)).pack(side="left",padx=8)
        ttk.Button(b,text="開啟輸出資料夾",command=self.openout).pack(side="right")
        self.after(100,self.poll)
    def picksrc(self):
        p=filedialog.askdirectory()
        if p:self.src.set(p);self.dst.set(self.dst.get() or str(Path(p).parent/"AI合照篩選結果_v1.2"))
    def pickdst(self):
        p=filedialog.askdirectory()
        if p:self.dst.set(p)
    def start(self):
        if not Path(self.src.get()).is_dir():messagebox.showwarning("提示","請選擇來源照片資料夾");return
        self.stop=False;self.go.config(state="disabled");threading.Thread(target=self.scan,daemon=True).start()
    def scan(self):
        try:
            src,dst=Path(self.src.get()),Path(self.dst.get());dst.mkdir(parents=True,exist_ok=True)
            score={"精準優先":.75,"平衡":.60,"團體照優先":.45}[self.sens.get()]
            det=cv2.FaceDetectorYN.create(str(model_path()),"",(320,320),score,.3,5000)
            mode=self.fm.get();limit={"完整分類":0,"只找合照：3人以上":3,"只找團體照：6人以上":6,"只找大合照：10人以上":10}[mode]
            if limit==0:
                for n,_ in CATS:(dst/n).mkdir(exist_ok=True)
            else:
                outdir=dst/{3:"只找合照_3人以上",6:"只找團體照_6人以上",10:"只找大合照_10人以上"}[limit];outdir.mkdir(exist_ok=True)
            files=[p for p in src.glob("**/*" if self.rec.get() else "*") if p.is_file() and p.suffix.lower() in EXTS]
            rows=[];selected=0
            for i,p in enumerate(files,1):
                if self.stop:break
                try:
                    im=image(p);h,w=im.shape[:2];det.setInputSize((w,h));_,faces=det.detect(im);n=0 if faces is None else len(faces)
                    cat=next(a for a,fn in CATS if fn(n));out=""
                    if limit==0 or n>=limit:
                        d=(dst/cat) if limit==0 else outdir;x=unique(d,p);shutil.copy2(p,x);out=str(x);selected+=1
                    rows.append([str(p),p.name,n,cat,out,"已輸出" if out else "未達門檻"])
                except Exception as e:rows.append([str(p),p.name,"","辨識失敗","",str(e)])
                if i%5==0 or i==len(files):self.q.put(("p",100*i/max(1,len(files)),f"處理中 {i}/{len(files)}"))
            with open(dst/"AI篩選結果_v1.2.csv","w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f);w.writerow(["原始路徑","檔名","人臉數","分類","輸出路徑","結果"]);w.writerows(rows)
            self.q.put(("done",f"完成！模式：{mode}\n共掃描 {len(files)} 張，輸出 {selected} 張。"))
        except Exception as e:self.q.put(("err",str(e)))
    def poll(self):
        try:
            while True:
                x=self.q.get_nowait()
                if x[0]=="p":self.p.set(x[1]);self.status.config(text=x[2])
                elif x[0]=="done":self.go.config(state="normal");messagebox.showinfo("完成",x[1])
                elif x[0]=="err":self.go.config(state="normal");messagebox.showerror("錯誤",x[1])
        except queue.Empty:pass
        self.after(100,self.poll)
    def openout(self):
        if Path(self.dst.get()).exists():os.startfile(self.dst.get())
if __name__=="__main__":App().mainloop()
