import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image
import numpy as np
import cv2
from ctypes import WinDLL
import ctypes
import os
from multiprocessing import Pool
import time
from scipy.io import wavfile
from scipy import signal
import subprocess
import moviepy.editor as mp

rgbhuetransform = WinDLL("./rgbhuetransform.so")
rgbhuetransform.TransformImageHSV.argtypes = np.ctypeslib.ndpointer(dtype=ctypes.c_int), ctypes.c_int, ctypes.c_int,ctypes.c_bool
rgbhuetransform.TransformImageHSV.restype = None
rgbhuetransform.TransformImageHSL.argtypes = np.ctypeslib.ndpointer(dtype=ctypes.c_int), ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,ctypes.c_int
rgbhuetransform.TransformImageHSL.restype = None
rgbhuetransform.LinearAdd.argtypes = np.ctypeslib.ndpointer(dtype=ctypes.c_int), np.ctypeslib.ndpointer(dtype=ctypes.c_int), ctypes.c_int, ctypes.c_float, ctypes.c_bool,ctypes.c_bool
rgbhuetransform.LinearAdd.restype = None
rgbhuetransform.AudioFormatter.argtypes = np.ctypeslib.ndpointer(dtype=ctypes.c_float), np.ctypeslib.ndpointer(dtype=ctypes.c_float), np.ctypeslib.ndpointer(dtype=ctypes.c_float), np.ctypeslib.ndpointer(dtype=ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
rgbhuetransform.AudioFormatter.restype = None

class ReactiveRGB:
    #files and file accesories
    backgroundFile = None #main image
    changeAreaFile = None #change base colour
    changeAreaMask = False #whether to change the main area or replace it
    glowAreaFile = None #add colour on top of
    audio = None #audio file

    #imagedata
    backgroundData = None
    changeAreaData= None
    glowAreaData= None

    changeAreaBlurredData = None


    #settings
    frameRate:int = 30
    rainbowRate:int = 4 #number of seconds on average to do a full colour rotation
    changeAreaGlowMin:int = 10      #min change area opacity
    changeAreaGlowMax:int = 100     #max change area opacity
    changeAreaGlowRadius:int = 20   #gaussian blur radius
    changeAreaGlowBase:int = 50    #percent of glow
    changeAreaGlowLinAdd:int = 30   #percent of linadd
    glowAreaMin:int = 0             #min opacity of glow
    glowAreaMax:int = 100           #max opacity of glow
    dbPercentileFloor:int = 10      #percentage of frames considered 0
    dbPercentileCeiling:int = 90    #percentage of frames considered 100
    glowMaxChange:int = 10          #highest amount of glow change from frame to frame
    glow2MaxChange:int = 10         #highest amount of glow2 change from frame to frame
    boomMaxChange:int=10            #highest amount of boom change from frame to frame
    threadCount:int = 20             #max number of threads to use 
    maxRAM:int = 12                 #max amount of ram usage, in GB

    eqRainbow:list = [100,100,100,100,100,100,100,100,100,100]
    eqGlow:list = [100,100,100,100,100,100,100,100,100,100]
    eqGlow2:list = [100,100,100,100,100,100,100,100,100,100]
    eqBoom:list = [100,75,50,25,0,0,0,0,0,0]

    EQFREQS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    def __init__(self):
        if not os.path.exists("temp"):
            os.makedirs("temp")
        if not os.path.exists("output"):
            os.makedirs("output")

    def preProcessImageFile(self,filename):
        newImage = Image.open(filename)
        if newImage.has_transparency_data:
            return newImage
        output = Image.new("RGBA",newImage.size)
        output.paste(newImage)
        return output

    def setBackground(self, filename):
        self.backgroundFile = filename
        self.backgroundData = self.preProcessImageFile(filename)

    def setChangeArea(self, filename):
        self.changeAreaFile = filename
        if self.changeAreaMask:
            self.changeAreaData = Image.new("RGBA",size = self.backgroundData.size)
            mask = self.preProcessImageFile(filename)
            mask = mask.convert("L")
            self.changeAreaData.paste(self.backgroundData, mask = mask)
        else:
            self.changeAreaData = self.preProcessImageFile(filename)

    def setChangeAreaMask(self, setting):
        self.changeAreaMask = setting
        self.setChangeArea(self.changeAreaFile)

    def setGlowArea(self, filename):
        self.glowAreaFile = filename
        self.glowAreaData = self.preProcessImageFile(filename)

    def setAudio(self, filename):
        self.audio = filename

    def setFrameRate(self,val:int):
        self.frameRate = int(val)
    def setRainbowRate(self,val:int):
        self.rainbowRate = int(val)
    def setChangeAreaGlowMin(self,val:int):
        self.changeAreaGlowMin = int(val)
    def setChangeAreaGlowMax(self,val:int):
        self.changeAreaGlowMax = int(val)
    def setChangeAreaGlowRadius(self,val:int):
        self.changeAreaGlowRadius = int(val)
    def setChangeAreaGlowBase(self,val:int):
        self.changeAreaGlowBase = int(val)
    def setChangeAreaGlowLinAdd(self,val:int):
        self.changeAreaGlowLinAdd = int(val)
    def setGlowAreaMin(self,val:int):
        self.glowAreaMin = int(val)
    def setGlowAreaMax(self,val:int):
        self.glowAreaMax = int(val)
    def setDbPercentileFloor(self,val:int):
        self.dbPercentileFloor = int(val)
    def setDbPercentileCeiling(self,val:int):
        self.dbPercentileCeiling = int(val)

    def setGlowMaxChange(self,val:int):
        self.glowMaxChange = int(val)
    def setGlow2MaxChange(self,val:int):
        self.glow2MaxChange = int(val)
    def setBoomMaxChange(self,val:int):
        self.boomMaxChange = int(val)


    def setThreadCount(self,val:int):
        self.threadCount = int(val)
    def setMaxRAM(self,val:int):
        self.maxRAM = int(val)
    def setEqGlow(self,vals:list):
        self.eqGlow = vals
    def setEqGlow(self, pos:int, val:int):
        self.eqGlow[pos] = int(val)
    def setEqGlow2(self,vals:list):
        self.eqGlow2 = vals
    def setEqGlow2(self, pos:int, val:int):
        self.eqGlow2[pos] = int(val)
    def setEqBoom(self,vals:list):
        self.eqBoom = vals
    def setEqBoom(self, pos:int, val:int):
        self.eqBoom[pos] = int(val)
    def setEqRainbow(self,vals:list):
        self.eqRainbow = vals
    def setEqRainbow(self, pos:int, val:int):
        self.eqRainbow[pos] = int(val)

class Frame():
    hue = 0
    glow = 100
    boom = 0
    wobble = 0
    tilt = 0

    def __init__(self, hue:int=0,glow:int=100,boom:int=0,wobble:int=0,tilt:int=0) -> None:
        self.hue = int(hue%360)
        
        if glow>100:
            glow = 100
        elif glow<0:
            glow = 0
        self.glow = int(glow)
        self.boom = int(boom)
        self.wobble = int(wobble)
        self.tilt = int(tilt)
    def __str__(self) -> str:
        return f'h{self.hue}g{self.glow}b{self.boom}w{self.wobble}t{self.tilt}'
    def setGlow(self, val):
        val = int(val)
        if val>100:val=100
        elif val<0:val=0
        self.glow = val


class AudioData():
    project:ReactiveRGB = None
    audioData = None
    # audioSorted = None
    totals = None
    frameCount = None
    runningTotals = None
    glowSorted = None
    glow2Sorted = None
    boomSorted = None

    def __init__(self, project:ReactiveRGB):
        self.project = project
        if self.project.audio.split(".")[-1].lower() == "wav":
            sf, audio = wavfile.read(self.project.audio)
        else:
            subprocess.call(['ffmpeg', '-i', self.project.audio,
                    'temp/tempaudio.wav'])
            sf, audio = wavfile.read('temp/tempaudio.wav')
            os.remove('temp/tempaudio.wav') 

        sig = np.mean(audio, axis=1)
        npts = int(sf/self.project.frameRate)
        f, t, Sxx = signal.spectrogram(sig, sf, nperseg=npts,nfft=npts*4)

        self.frameCount = int(t[-2]*self.project.frameRate)
        adata = np.zeros((self.frameCount,10), dtype=np.single)
        f = np.single(f)
        t = np.single(t)
        Sxx = np.single(Sxx)
        rgbhuetransform.AudioFormatter(adata, Sxx, f, t, self.frameCount, len(t), len(f), self.project.frameRate)

        self.audioData = adata.astype(float)
        self.totals = [0,0,0,0,0,0,0,0,0,0]
        self.runningTotals = []
        print(self.audioData.shape)
        for row in range(self.audioData.shape[0]):
            thisLine = []
            for col in range(self.audioData.shape[1]):
                self.totals[col] = self.totals[col] + self.audioData[row][col]
                thisLine.append(self.totals[col])
            self.runningTotals.append(thisLine)
        # self.audioSorted = self.audioData.copy()
        # sidx = self.audioSorted.argsort(axis=0)
        # self.audioSorted = self.audioSorted[sidx, np.arange(sidx.shape[1])]
        self.glowSorted = []
        self.glow2Sorted = []
        self.boomSorted = []
        total = 0.001
        totalweight =0.001
        for row in range(self.audioData.shape[0]):
            total = 0.001
            totalweight =0.001
            for freq in range(10):
                total = total + self.project.eqGlow[freq]*self.audioData[row][freq]
                totalweight = totalweight + self.project.eqGlow[freq]
            self.glowSorted.append(total/totalweight)
            total = 0.001
            totalweight =0.001
            for freq in range(10):
                total = total + self.project.eqGlow2[freq]*self.audioData[row][freq]
                totalweight = totalweight + self.project.eqGlow2[freq]
            self.glow2Sorted.append(total/totalweight)
            total = 0.001
            totalweight =0.001
            for freq in range(10):
                total = total + self.project.eqBoom[freq]*self.audioData[row][freq]
                totalweight = totalweight + self.project.eqBoom[freq]
            self.boomSorted.append(total/totalweight)
        self.glowSorted.sort()
        self.glow2Sorted.sort()
        self.boomSorted.sort()
        with open("glowIntensity.csv","w") as f:
            for line in self.glowSorted:
                f.write(f'{line}\n')

    def hueProgression(self,frame)->int:
        total = 0.1
        totalweight = 0.1
        for freq in range(10):
            total = total + self.project.eqRainbow[freq]*self.runningTotals[frame][freq]/self.totals[freq]
            totalweight = totalweight + self.project.eqRainbow[freq]
        hue= ((self.frameCount/self.project.frameRate/self.project.rainbowRate)*360*(total/totalweight))%360
        return hue
    
    def glow(self,frame)->int:
        total = 0.1
        totalweight = 0.1
        for freq in range(10):
            total = total + self.project.eqGlow[freq]*self.audioData[frame][freq]
            totalweight = totalweight + self.project.eqGlow[freq]

        glow = self.project.changeAreaGlowMin + (self.project.changeAreaGlowMax - self.project.changeAreaGlowMin) * (total/totalweight-self.glowSorted[int(len(self.glowSorted)*self.project.dbPercentileFloor/100)])/(self.glowSorted[int(len(self.glowSorted)*self.project.dbPercentileCeiling/100)]-self.glowSorted[int(len(self.glowSorted)*self.project.dbPercentileFloor/100)]) 
        return int(glow)

def preProcessStack(project:ReactiveRGB):
    if project.changeAreaFile:
        project.changeAreaBlurredData = Image.fromarray(cv2.blur(np.array(project.changeAreaData),(project.changeAreaGlowRadius,project.changeAreaGlowRadius)))
    
def processFrame(project:ReactiveRGB, frame:Frame)-> Image:
    # print("--------------------------")
    # t = time.time_ns()
    newImage = project.backgroundData.copy()
    if project.changeAreaData:
        # blurred part
        changearea = shiftColour(project.changeAreaBlurredData,frame.hue)
        newblur = project.backgroundData.copy()
        newblur.paste(changearea,mask=changearea)
        newImage = Image.blend(newImage, newblur, alpha=(project.changeAreaGlowBase * (project.changeAreaGlowMin + frame.glow*float(project.changeAreaGlowMax - project.changeAreaGlowMin)/100)/10000))
        
        # regular part
        changearea = shiftColour(project.changeAreaData,frame.hue)
        newImage.paste(changearea,mask=changearea)

        #linear
        changearea = shiftColour(project.changeAreaBlurredData,frame.hue)
        alpha=project.changeAreaGlowLinAdd * (project.changeAreaGlowMin + frame.glow*float(project.changeAreaGlowMax - project.changeAreaGlowMin)/100)/10000
        newImage = linearAdd(newImage,changearea,alpha)
        
    if project.glowAreaData:
        glowarea = shiftColour(project.glowAreaData,frame.hue)
        alpha=(project.glowAreaMin + frame.glow*float(project.glowAreaMax - project.glowAreaMin)/100)/100
        newImage = linearAdd(newImage,glowarea,alpha)

    # print(time.time_ns()-t)
    return newImage

def threadProcessFrame(things):
    project, frames = things
    output = []
    for frame in frames:
        output.append([frame[0], processFrame(project, frame[1])])
    return output
def tempSave(imgandname):
    imgandname[0].save(f"./temp/{imgandname[1]}.png")
#HSL version
def shiftColour(image:Image, hueShift:float, saturationShift:float=0.0, luminanceShift = 0.0)->Image:
    img = np.asarray(image, dtype=np.int32)
    # print(img.shape[0]*img.shape[1])
    # print(img[0][0])
    rgbhuetransform.TransformImageHSL(img,int(img.shape[0]*img.shape[1]), hueShift,saturationShift,luminanceShift, len(img[0][0]))
    return Image.fromarray(np.uint8(img))

def linearAdd(img:Image, imgadd:Image, alpha:float)->Image:
    if alpha>1:alpha=1
    elif alpha<0:alpha=0
    imgArr = np.asarray(img, dtype=np.int32)
    imgAddArr = np.asarray(imgadd, dtype=np.int32)
    rgbhuetransform.LinearAdd(imgArr,imgAddArr,int(imgArr.shape[0]*imgArr.shape[1]),alpha, img.has_transparency_data, imgadd.has_transparency_data)

    return Image.fromarray(np.uint8(imgArr))


def preview(project:ReactiveRGB):
    processFrame(project,Frame(hue=0)).save("rainbowoutput1.png")
    processFrame(project,Frame(hue=85)).save("rainbowoutput2.png")
    processFrame(project,Frame(hue=170)).save("rainbowoutput3.png")

def render(project:ReactiveRGB):
    t =time.time_ns()
    p = Pool(project.threadCount)
    preProcessStack(project)

    frameOrder = []
    frames = {}
    if project.audio:
        audioData = AudioData(project)
        frameCount = audioData.frameCount
        print("making frames")
        # maxGlow = 0
        # maxGlow2 = 0
        # maxBoom = 0
        lastGlow = 0
        for f in range(frameCount):
            hue = audioData.hueProgression(f)
            glow=audioData.glow(f)
            if lastGlow+project.glowMaxChange<glow: glow = lastGlow+project.glowMaxChange
            elif lastGlow-project.glowMaxChange>glow: glow = lastGlow-project.glowMaxChange
            lastGlow = glow
            # if glow>maxGlow: maxGlow=glow
            newFrame = Frame(hue, glow = glow)
            # print(newFrame)
            frameOrder.append(newFrame)
        for f in range(len(frameOrder)):
            # frameOrder[f].setGlow(100*frameOrder[f].glow/maxGlow)
            if frameOrder[f].__str__() not in frames:
                frames[frameOrder[f].__str__()] = {"frame":frameOrder[f],"num":0,"hasFile":False}
            frames[frameOrder[f].__str__()]['num']+=1
            frameOrder[f]=str(frameOrder[f])
        

    else:
        frameCount = project.frameRate*project.rainbowRate
        for f in range(frameCount):
            newFrame = Frame(hue=f*360/frameCount)
            frameOrder.append(newFrame.__str__())
            if newFrame.__str__() not in frames:
                frames[newFrame.__str__()] = {"frame":newFrame,"num":0,"hasFile":False}
            frames[newFrame.__str__()]['num']+=1
    print("frames made")
    if project.audio:
        vidname = f'./output/temp{time.time()}.mp4'
        finalvidname = f'./output/output{time.time()}.mp4'
    else:
        vidname = f'./output/output{time.time()}.mp4'
    video = cv2.VideoWriter(vidname,0,project.frameRate,project.backgroundData.size)
    batchSize = project.maxRAM*1000000000/(project.backgroundData.size[0]*project.backgroundData.size[1]*4*4)
    batchCount = 0
    

    frameNum = 0
    workFrameNum = 0
    while batchSize*batchCount<frameCount:
        #prepping the frame work list list
        thisBatchSize = 0
        frameWork = []
        framesToDo = set()
        for i in range(project.threadCount):
            frameWork.append([project,[]])
        nextThread = 0
        while thisBatchSize<batchSize and workFrameNum<len(frameOrder):

            if not frames[frameOrder[workFrameNum]]["hasFile"] and frameOrder[workFrameNum] not in framesToDo:
                frameWork[nextThread][1].append([frameOrder[workFrameNum],frames[frameOrder[workFrameNum]]['frame']])
                framesToDo.add(frameOrder[workFrameNum])
                thisBatchSize+=1
                nextThread=(nextThread+1)%project.threadCount
            workFrameNum+=1


        #process all the new frames
        output = p.map(threadProcessFrame,frameWork)

        newFrames = {}
        for frameSet in output:
            for frame in frameSet:
                newFrames[frame[0]]=frame[1]
        
        while(frameNum<workFrameNum):
            #get images either from the new processed images or from disk
            if frameOrder[frameNum] in newFrames:
                thisFrame = np.asarray(newFrames[frameOrder[frameNum]], dtype=np.uint8)
            else:
                thisFrame = np.asarray(Image.open(f".temp/{frameOrder[frameNum]}.png"), dtype=np.uint8)

            #WHY DOES CV2 USE BGR
            thisFrame = cv2.cvtColor(thisFrame, cv2.COLOR_RGB2BGR)
            video.write(thisFrame)

            frames[frameOrder[frameNum]]["num"]-=1
            frameNum+=1
        #if any images will be used in the future, save them
        savelist = []
        for frame in newFrames.keys():
            if frames[frame]["num"]>0 and not frames[frame]["hasFile"]:
                savelist.append([newFrames[frame], frame])
        p.map(tempSave, savelist)
                
        
        batchCount+=1

    video.release()
    
    if project.audio:
        with mp.VideoFileClip(vidname) as video:
            audio = mp.AudioFileClip(project.audio)
            video = video.set_audio(audio)
            video.write_videofile(finalvidname)
        os.remove(vidname) 

    #remove all temp files
    for f in [os.path.join("./temp",f) for f in os.listdir("./temp")]:
        os.remove(f) 

    print(f"TIME: {(time.time_ns()-t)/1000000}ms")


def UI(project:ReactiveRGB = None):
    if project is None: project = ReactiveRGB()
    ui = tk.Tk()
    ui.title('Rainbowing Audio')
    
    

    #images
    # minImage = tk.Label( ui ,height= 20)
    # midImage = tk.Label(ui ,height= 20)
    # maxImage = tk.Label(ui ,height= 20)
    # minImage.grid(column = 1)
    # midImage.grid(column = 1)
    # maxImage.grid(column = 1)

    #mainbuttons
    backgroundButton = tk.Button(ui, text='Background', width=25, command=lambda:project.setBackground(askopenfilename()))
    changeAreaButton = tk.Button(ui, text='Changing Area', width=25, command=lambda:project.setChangeArea(askopenfilename()))
    glowAreaButton = tk.Button(ui, text='Glow Area', width=25, command=lambda:project.setGlowArea(askopenfilename()))
    audioFileButton = tk.Button(ui, text='Audio', width=25, command=lambda:project.setAudio(askopenfilename()))
    setButton = tk.Button(ui, text="PREVIEW", command=lambda:preview(project))
    renderButton = tk.Button(ui, text="RENDER", command=lambda:render(project))
    # setButton = tk.Button(ui, text="SET", command=lambda:previewImage(minImage,midImage,maxImage,project))
    backgroundButton.grid(row = 0, column = 0)
    changeAreaButton.grid(row = 1, column = 0)
    glowAreaButton.grid(row = 2, column = 0)
    audioFileButton.grid(row = 3, column = 0)
    setButton.grid(row = 4, column = 0)
    renderButton.grid(row = 5, column = 0)


    #sliders
    sliders = [
        # [label, description, start value, lambda, min, max]
        ["Frame Rate","",project.frameRate,lambda val:project.setFrameRate(int(val)),1,100],
        ["Rainbow Rate","Average number of seconds per rainbow rotation",project.rainbowRate,lambda val:project.setRainbowRate(int(val)),0,100],
        ["Minimum Glow","min change area opacity",project.changeAreaGlowMin,lambda val:project.setChangeAreaGlowMin(int(val)),0,100],
        ["Maximum Glow","max change area opacity",project.changeAreaGlowMax,lambda val:project.setChangeAreaGlowMax(int(val)),0,100],
        ["Glow Radius","gaussian blur radius on glow",project.changeAreaGlowRadius,lambda val:project.setChangeAreaGlowRadius(int(val)) ,0,100],
        ["Change Area Glow","Base Change Area Glow",project.changeAreaGlowBase,lambda val:project.setChangeAreaGlowBase(int(val)),0,100],
        ["","",project.changeAreaGlowLinAdd,lambda val:project.setChangeAreaGlowLinAdd(int(val)),0,100],
        ["2nd Glow Area Min","",project.glowAreaMin,lambda val:project.setGlowAreaMin(int(val)) ,0,100],
        ["2nd Glow Area Max","",project.glowAreaMax,lambda val:project.setGlowAreaMax(int(val)),0,100],
        ["db floor","",project.dbPercentileFloor,lambda val:project.setDbPercentileFloor(int(val)) ,0,100],
        ["db ceiling","",project.dbPercentileCeiling,lambda val:project.setDbPercentileCeiling(int(val)),0,100],
        ["max glow change","",project.glowMaxChange,lambda val:project.setGlowMaxChange(int(val)) ,0,100],
        ["glow2 max change","",project.glow2MaxChange,lambda val:project.setGlow2MaxChange(int(val)),0,100],
        ["boom max change","",project.boomMaxChange,lambda val:project.setBoomMaxChange(int(val)),0,100],
        ["thread count","",project.threadCount,lambda val:project.setThreadCount(int(val)),1,64],
        ["max RAM (GB)","THIS IS AN ESTIMATE",project.maxRAM,lambda val:project.setMaxRAM(int(val)),1,64]
    ]
    
    sliderObjects = []
    counter = 1
    for slider in sliders:
        newSlider = tk.Scale(ui, from_=slider[4], to=slider[5], orient=tk.HORIZONTAL, command= slider[3] )
        newSlider.grid(row = counter, column = 4)
        newLabel = tk.Label(text=slider[0])
        newLabel.grid(row =  counter, column=  3)
        newDescr = tk.Label(text=slider[1])
        newDescr.grid(row =  counter, column=  5)
        newSlider.set(slider[2])
        sliderObjects.append([newSlider,newLabel,newDescr])
        counter+=2

    eqRainbowParts = []
    eqRainbowLabel = tk.Label(text="Rainbow Reactivity")
    eqRainbowLabel.grid(row =  1, column=  6)
    for n in range(len(project.EQFREQS)):
        eqRainbowParts.append({})
        eqRainbowParts[n]['slider'] = tk.Scale(ui, from_=100, to=0, orient=tk.VERTICAL, command= lambda val:project.setEqRainbow(n, int(val)) )
        eqRainbowParts[n]['slider'].grid(row = 1, column = 7+n)
        eqRainbowParts[n]['slider'].set(project.eqRainbow[n])
        eqRainbowParts[n]['label'] = tk.Label(text=project.EQFREQS[n])
        eqRainbowParts[n]['label'].grid(row =  2, column=  7+n)

    eqGlowParts = []
    eqGlowLabel = tk.Label(text="Glow Reactivity")
    eqGlowLabel.grid(row =  3, column=  6)
    for n in range(len(project.EQFREQS)):
        eqGlowParts.append({})
        eqGlowParts[n]['slider'] = tk.Scale(ui, from_=100, to=0, orient=tk.VERTICAL, command= lambda val:project.setEqGlow(n, int(val)) )
        eqGlowParts[n]['slider'].grid(row = 3, column = 7+n)
        eqGlowParts[n]['slider'].set(project.eqGlow[n])
        eqGlowParts[n]['label'] = tk.Label(text=project.EQFREQS[n])
        eqGlowParts[n]['label'].grid(row =  4, column=  7+n)

    eqGlow2Parts = []
    eqGlow2Label = tk.Label(text="Glow2 Reactivity")
    eqGlow2Label.grid(row =  5, column=  6)
    for n in range(len(project.EQFREQS)):
        eqGlow2Parts.append({})
        eqGlow2Parts[n]['slider'] = tk.Scale(ui, from_=100, to=0, orient=tk.VERTICAL, command= lambda val:project.setEqGlow2(n, int(val)) )
        eqGlow2Parts[n]['slider'].grid(row = 5, column = 7+n)
        eqGlow2Parts[n]['slider'].set(project.eqGlow2[n])
        eqGlow2Parts[n]['label'] = tk.Label(text=project.EQFREQS[n])
        eqGlow2Parts[n]['label'].grid(row =  6, column=  7+n)

    eqBoomParts = []
    eqBoomLabel = tk.Label(text="BOOM Reactivity")
    eqBoomLabel.grid(row =  7, column=  6)
    for n in range(len(project.EQFREQS)):
        eqBoomParts.append({})
        eqBoomParts[n]['slider'] = tk.Scale(ui, from_=100, to=0, orient=tk.VERTICAL, command= lambda val:project.setEqBoom(n, int(val)) )
        eqBoomParts[n]['slider'].grid(row = 7, column = 7+n)
        eqBoomParts[n]['slider'].set(project.eqBoom[n])
        eqBoomParts[n]['label'] = tk.Label(text=project.EQFREQS[n])
        eqBoomParts[n]['label'].grid(row =  8, column=  7+n)


    ui.mainloop()
if __name__ == "__main__":
    UI()