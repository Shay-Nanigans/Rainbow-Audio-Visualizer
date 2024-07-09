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
import yaml
import winsound
import pickle
# import moviepy.editor as mp

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

    layers = {}
    layercounter = 0

    #settings
    config={}

    EQFREQS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    def __init__(self):
        if not os.path.exists("temp"):
            os.makedirs("temp")
        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("config.txt"):
            os.popen('copy defaultconfig.txt config.txt')
            time.sleep(1)
        self.loadConfig()
        
        print(self.config)


    def setBackground(self, filename):
        self.backgroundFile = filename
        self.backgroundData = Image.open(filename).convert('RGBA')

    def setAudio(self, filename):
        self.audio = filename

    def loadConfig(self):
        with open("config.txt") as f:
            self.config = yaml.safe_load(f)

    def saveConfig(self):
        with open("config.txt", "w") as f:
            yaml.safe_dump(self.config, f)
    def resetConfig(self):
        os.popen('copy defaultconfig.txt config.txt')
        time.sleep(1)
        self.loadConfig()

    def newLayer(self, file:str):
        if file:
            self.layers[self.layercounter] = RainbowLayer(self.layercounter,self, file)
            self.layercounter+=1

    def setEqBoom(self,vals:list):
        self.config["eqBoom"] = vals
    def setEqBoom(self, pos:int, val:int):
        self.config["eqBoom"][pos] = int(val)

    def setConfig(self,key:str,val:int):
        self.config[key] = val
    def setConfigInt(self,key:str,val:int):
        self.config[key] = int(val)

    def save(self, filename):
        data = {"backgroundFile":self.backgroundFile,
        "changeAreaFile":self.changeAreaFile,
        "changeAreaMask":self.changeAreaMask,
        "glowAreaFile":self.glowAreaFile,
        "audio": self.audio,
        "layers":self.layers,
        "layercounter":self.layercounter,
        "config":self.config,
                }
        open(filename,"wb").write(pickle.dumps(data))
    
    def load(self, filename):
        data = pickle.load(open(filename,"rb"))
        self.backgroundFile=data["backgroundFile"]
        self.changeAreaFile=data["changeAreaFile"]
        self.changeAreaMask=data["changeAreaMask"]
        self.glowAreaFile=data["glowAreaFile"]
        self.audio=data["audio"]
        self.layers=data["layers"]
        self.layercounter=data["layercounter"]
        self.config=data["config"]
        self.setBackground(self.backgroundFile)

    def cleanup(self):
        for layer in self.layers:
            layer.cleanup()

class RainbowLayer():
    layerID:int = None
    project:ReactiveRGB = None
    imgFile:str = None
    imgMask:str = 'PNG'
    config:dict = None

    imgData = None
    imgBlurredData=None
    glowData = None
    def __init__(self, layerID:int, project:ReactiveRGB,file:str):
        self.imgFile = file
        self.layerID = layerID
        self.project = project
        self.reset()

    def prepImg(self):
        if self.imgMask == 'PNG':
            self.imgData = Image.open(self.imgFile).convert('RGBA')
        elif self.imgMask == 'BASE':
            self.imgData = Image.new("RGBA",size = self.project.backgroundData.size)
            mask = Image.open(self.imgFile).convert('RGBA')
            mask = mask.convert("L")
            self.imgData.paste(self.project.backgroundData, mask = mask)
        else:
            self.imgData = Image.new("RGBA",size = self.project.backgroundData.size)
            mask = Image.open(self.imgFile).convert('RGBA')
            mask = mask.convert("L")
            baseImg = Image.open(self.imgMask).convert('RGBA')
            self.imgData.paste(baseImg, mask = mask)
    def reset(self):
        self.config = self.project.config['defaultlayer'].copy()
        self.config['eqRainbow'] = self.project.config['defaultlayer']['eqRainbow'].copy()
        self.config['eqGlow'] = self.project.config['defaultlayer']['eqGlow'].copy()
    def setMask(self,val):
        self.imgMask = str(val)
    def setFile(self,val):
        self.imgFile = str(val)
    
    def setConfig(self,key:str,val):
        self.config[key] = val
    def setConfigInt(self,key:str,val:int):
        self.config[key] = int(val)

    def setEq(self,key, vals:list):
        self.config[key] = vals
    def setEq(self,key,  pos:int, val:int):
        self.config[key][pos] = int(val)

    def cleanup(self):
        self.imgData = None
        self.glowData = None
        self.rainbowData = None

class Frame():
    hue = None
    glow = None
    boom = 0
    wobble = 0
    tilt = 0

    def __init__(self, hue:list,glow:list,boom:int=0,wobble:int=0,tilt:int=0) -> None:
        self.hue = []
        self.glow = []
        for h in hue:
            self.hue.append(int(h%360))
        for g in glow:
            if g>100:
                g = 100
            elif g<0:
                g = 0
            self.glow.append(int(g))
        if boom>100:
            boom = 100
        elif boom<0:
            boom = 0
        self.boom = int(boom)
        self.wobble = int(wobble)
        self.tilt = int(tilt)
    def __str__(self) -> str:
        id = ""
        for h in self.hue:
            id=f'{id}h{h}'
        for g in self.glow:
            id = f'{id}g{g}'
        return f'{id}b{self.boom}w{self.wobble}t{self.tilt}'
    def setGlow(self, glow):
        for g in glow:
            if g>100:
                g = 100
            elif g<0:
                g = 0
            self.glow.append(int(glow))


class AudioData():
    project:ReactiveRGB = None
    audioData = None
    # audioSorted = None
    totals = None
    frameCount = None
    runningTotals = None
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
        npts = int(sf/self.project.config["frameRate"])
        f, t, Sxx = signal.spectrogram(sig, sf, nperseg=npts,nfft=npts*4)

        self.frameCount = int(t[-2]*self.project.config["frameRate"])
        adata = np.zeros((self.frameCount,10), dtype=np.single)
        f = np.single(f)
        t = np.single(t)
        Sxx = np.single(Sxx)
        rgbhuetransform.AudioFormatter(adata, Sxx, f, t, self.frameCount, len(t), len(f), self.project.config["frameRate"])

        self.audioData = adata.astype(float)
        # np.savetxt("audios.csv", self.audioData, delimiter=",") #sometimes a gal's gotta just see the data
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

        for layer in project.layers.keys():        
            glowSorted = []
            total = 0
            totalweight =0
            for row in range(self.audioData.shape[0]):
                total = 0
                totalweight =0
                for freq in range(10):
                    total = total + self.project.layers[layer].config["eqGlow"][freq]*self.audioData[row][freq]
                    totalweight = totalweight + self.project.layers[layer].config["eqGlow"][freq]
                glowSorted.append(total/totalweight)
            glowSorted.sort()  
            self.project.layers[layer].glowData=glowSorted
                
        self.boomSorted = []    
        for row in range(self.audioData.shape[0]):
            total = 0
            totalweight =0
            for freq in range(10):
                total = total + self.project.config["eqBoom"][freq]*self.audioData[row][freq]
                totalweight = totalweight + self.project.config["eqBoom"][freq]
            self.boomSorted.append(total/totalweight)
        # with open("glowIntensityunsort.csv","w") as f:
        #     for line in self.glowSorted:
        #         f.write(f'{line}\n')
        
        self.boomSorted.sort()
        # with open("glowIntensity.csv","w") as f:
        #     for line in self.glowSorted:
        #         f.write(f'{line}\n')
        
        self.boomProcessed = []
        for i in range(self.audioData.shape[0]):
            self.boomProcessed.append(self.boom(i))

        if self.project.config["boomwinlen"]>1:  
            self.boomProcessed = signal.savgol_filter(self.boomProcessed,self.project.config["boomwinlen"],self.project.config["boompolyorder"])
            self.boomProcessed = self.boomProcessed.tolist()
            for i in range(abs(self.project.config["boomoffset"])):
                if self.project.config["boomoffset"]>0:
                    self.boomProcessed.insert(0,0)
                    self.boomProcessed.pop()
                elif self.project.config["boomoffset"]<0:
                    self.boomProcessed.append(0)
                    self.boomProcessed.pop(0)
        self.boomProcessed =rescaleList(self.boomProcessed,0,100,True)
                

    def hueProgression(self,frame,layer)->int:
        if self.project.layers[layer].config["rainbowRate"]==0: return 0
        total = 0.0
        totalweight = 0.0
        for freq in range(10):
            total = total + self.project.layers[layer].config["eqRainbow"][freq]*self.runningTotals[frame][freq]/self.totals[freq]
            totalweight = totalweight + self.project.layers[layer].config["eqRainbow"][freq]
        hue= ((self.frameCount/self.project.config["frameRate"]/self.project.layers[layer].config["rainbowRate"])*360*(total/totalweight))%360
        return hue
    
    def glow(self,frame,layer)->int:
        total = 0.0
        totalweight = 0.0
        for freq in range(10):
            total = total + self.project.layers[layer].config["eqGlow"][freq]*self.audioData[frame][freq]
            totalweight = totalweight + self.project.layers[layer].config["eqGlow"][freq]

        if totalweight==0 or total==0: return 0

        glow = self.project.layers[layer].config["changeAreaGlowMin"] + (self.project.layers[layer].config["changeAreaGlowMax"] - self.project.layers[layer].config["changeAreaGlowMin"]) * (total/totalweight-self.project.layers[layer].glowData[int(len(self.project.layers[layer].glowData)*self.project.config["dbPercentileFloor"]/100)])/(self.project.layers[layer].glowData[int(len(self.project.layers[layer].glowData)*self.project.config["dbPercentileCeiling"]/100)]-self.project.layers[layer].glowData[int(len(self.project.layers[layer].glowData)*self.project.config["dbPercentileFloor"]/100)]) 
        if glow<0:glow=0
        elif glow>100:glow=0
        return int(glow)
    
    def boom(self,frame)->int:
        total = 0.0
        totalweight = 0.0
        for freq in range(10):
            total = total + self.project.config["eqBoom"][freq]*self.audioData[frame][freq]
            totalweight = totalweight + self.project.config["eqBoom"][freq]

        if totalweight==0 or total==0: return 0
        boom = (total/totalweight-self.boomSorted[int(len(self.boomSorted)*self.project.config["dbPercentileFloor"]/100)])/(self.boomSorted[int(len(self.boomSorted)*self.project.config["dbPercentileCeiling"]/100)]-self.boomSorted[int(len(self.boomSorted)*self.project.config["dbPercentileFloor"]/100)])*100
        if boom<0:boom=0
        elif boom>100:boom=0
        return int(boom)

def rescaleList(things:list,newMin,newMax,isInt:bool = False):
    oldMax = max(things)
    oldMin = min(things)
    m = (newMax-newMin)/(oldMax-oldMin)
    b = newMin-oldMin*(newMax-newMin)/(oldMax-oldMin)
    for thing in range(len(things)):
        things[thing] = m*things[thing]+b
        if isInt:things[thing]=int(things[thing])
    return things

def preProcessStack(project:ReactiveRGB):
    for key in project.layers.keys():
        project.layers[key].prepImg()
        if project.layers[key].config["changeAreaGlowRadius"]>0:
            project.layers[key].imgBlurredData = Image.fromarray(cv2.blur(np.array(project.layers[key].imgData),(project.layers[key].config["changeAreaGlowRadius"],project.layers[key].config["changeAreaGlowRadius"])))
        else: 
            project.layers[key].imgBlurredData = project.layers[key].imgData 


def processFrame(project:ReactiveRGB, frame:Frame)-> Image:
    # print("--------------------------")
    # t = time.time_ns()
    newImage = project.backgroundData.copy()
    i=0
    for layer in project.layers.keys():

        # blurred part
        alpha=(project.layers[layer].config["changeAreaGlowBase"] * (project.layers[layer].config["changeAreaGlowMin"] + frame.glow[i]*float(project.layers[layer].config["changeAreaGlowMax"] - project.layers[layer].config["changeAreaGlowMin"])/100)/10000)
        if(alpha>0):
            changearea = shiftColour(project.layers[layer].imgBlurredData,frame.hue[i],project.layers[layer].config['saturationShift'],project.layers[layer].config['luminanceShift'])
            newblur = project.backgroundData.copy()
            newblur.paste(changearea,mask=changearea)
            newImage = Image.blend(newImage, newblur, alpha=(project.layers[layer].config["changeAreaGlowBase"] * (project.layers[layer].config["changeAreaGlowMin"] + frame.glow[i]*float(project.layers[layer].config["changeAreaGlowMax"] - project.layers[layer].config["changeAreaGlowMin"])/100)/10000))
        
        # regular part
        if project.layers[layer].config["hasBaseLayer"]:
            changearea = shiftColour(project.layers[layer].imgData,frame.hue[i],project.layers[layer].config['saturationShift'],project.layers[layer].config['luminanceShift'])
            newImage.paste(changearea,mask=changearea)

        #linear
        alpha=project.layers[layer].config["changeAreaGlowLinAdd"] * (project.layers[layer].config["changeAreaGlowMin"] + frame.glow[i]*float(project.layers[layer].config["changeAreaGlowMax"] - project.layers[layer].config["changeAreaGlowMin"])/100)/10000
        if(alpha>0):
            changearea = shiftColour(project.layers[layer].imgBlurredData,frame.hue[i],project.layers[layer].config['saturationShift'],project.layers[layer].config['luminanceShift'])
            newImage = linearAdd(newImage,changearea,alpha)
        i+=1
    if project.config["maxBoom"]>0 and frame.boom>0:
        scale = 1 + project.config["maxBoom"]*frame.boom/10000.0 
        imgArr = np.asarray(newImage)
        newArr = imgArr[int((newImage.size[1]-newImage.size[1]/scale)/2) :int((newImage.size[1]+newImage.size[1]/scale)/2), int((newImage.size[0]-newImage.size[0]/scale)/2) :int((newImage.size[0]+newImage.size[0]/scale)/2)]
        newImage = Image.fromarray(cv2.resize(newArr,( newImage.size[0],newImage.size[1])))

    # print(time.time_ns()-t)
    return newImage

def threadProcessFrame(things):
    project, frames, layers = things
    project.layers = layers
    output = []
    for frame in frames:
        output.append([frame[0], processFrame(project, frame[1])])
    return output
def tempSave(imgandname):
    imgandname[0].save(f"./temp/{imgandname[1]}.png")
#HSL version
def shiftColour(image:Image, hueShift:float, saturationShift:float=0.0, luminanceShift = 0.0)->Image:
    saturationShift = saturationShift/100
    luminanceShift = luminanceShift/100 

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
    rgbhuetransform.LinearAdd(imgArr,imgAddArr,int(imgArr.shape[0]*imgArr.shape[1]),alpha, True, True)

    return Image.fromarray(np.uint8(imgArr))

def PID(pid:list,pidsettings:list):
    current,target,errorlast,ierror = pid
    p,i,d = pidsettings
    error = target - current
    ierror+=error
    derror = error - errorlast
    output = p*error + i*ierror + d*derror
    return [int(output), target, error, ierror]

def preview(project:ReactiveRGB):
    processFrame(project,Frame(hue=0)).save("rainbowoutput1.png")
    processFrame(project,Frame(hue=85)).save("rainbowoutput2.png")
    processFrame(project,Frame(hue=170)).save("rainbowoutput3.png")

def render(project:ReactiveRGB):
    t =time.time_ns()
    p = Pool(project.config["threadCount"])

    #remove all temp files from any previous cancelled/crashed attempts
    for f in [os.path.join("./temp",f) for f in os.listdir("./temp")]:
        os.remove(f) 

    preProcessStack(project)
    frameOrder = []
    frames = {}
    if project.audio:
        audioData = AudioData(project)
        frameCount = audioData.frameCount
        print("making frames")

        lastGlow = []
        for layer in range(len(project.layers)):
            lastGlow.append(0)

        for f in range(frameCount):
            hue = []
            glow = []
            i=0
            for layer in project.layers.keys():
                hue.append(audioData.hueProgression(f,layer))
                glow.append(audioData.glow(f,layer))
                
                if lastGlow[i]+project.layers[layer].config["glowMaxIncrease"]<glow[i]: glow[i] = lastGlow[i]+project.layers[layer].config["glowMaxIncrease"]
                elif lastGlow[i]-project.layers[layer].config["glowMaxDecrease"]>glow[i]: glow[i] = lastGlow[i]-project.layers[layer].config["glowMaxDecrease"]
                lastGlow[i] = glow[i]
                i+=1
            if project.config["maxBoom"]>0: 
                boom = audioData.boomProcessed[f]
            else:
                boom = 0

            newFrame = Frame(hue, glow = glow,boom=boom)
            # print(newFrame)
            frameOrder.append(newFrame)
        for f in range(len(frameOrder)):
            # frameOrder[f].setGlow(100*frameOrder[f].glow/maxGlow)
            if frameOrder[f].__str__() not in frames:
                frames[frameOrder[f].__str__()] = {"frame":frameOrder[f],"num":0,"hasFile":False}
            frames[frameOrder[f].__str__()]['num']+=1
            frameOrder[f]=str(frameOrder[f])
        

    else:
        frameCount = project.config["frameRate"]*project.layers[list(project.layers.keys())[0]].config["rainbowRate"]
        for f in range(frameCount):
            newFrame = Frame(hue=f*360/frameCount)
            frameOrder.append(newFrame.__str__())
            if newFrame.__str__() not in frames:
                frames[newFrame.__str__()] = {"frame":newFrame,"num":0,"hasFile":False}
            frames[newFrame.__str__()]['num']+=1
    print("frames made")

    vidname = f'./temp/temp{time.time()}.mp4'
    finalvidname = f'./output/output{time.time()}.mp4'

    video = cv2.VideoWriter(vidname,0,project.config["frameRate"],project.backgroundData.size)
    batchSize = project.config["maxRAM"]*1000000000/(project.backgroundData.size[0]*project.backgroundData.size[1]*4*4)
    batchCount = 0
    

    frameNum = 0
    workFrameNum = 0
    while batchSize*batchCount<frameCount:
        #prepping the frame work list list
        thisBatchSize = 0
        frameWork = []
        framesToDo = set()
        for i in range(project.config["threadCount"]):
            frameWork.append([project,[],project.layers])
        nextThread = 0
        while thisBatchSize<batchSize and workFrameNum<len(frameOrder):

            if not frames[frameOrder[workFrameNum]]["hasFile"] and frameOrder[workFrameNum] not in framesToDo:
                frameWork[nextThread][1].append([frameOrder[workFrameNum],frames[frameOrder[workFrameNum]]['frame']])
                framesToDo.add(frameOrder[workFrameNum])
                thisBatchSize+=1
                nextThread=(nextThread+1)%project.config["threadCount"]
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
        # with mp.VideoFileClip(vidname) as video:
        #     audio = mp.AudioFileClip(project.audio)
        #     video = video.set_audio(audio)
        #     video.write_videofile(finalvidname)
        subprocess.call(["ffmpeg", "-i", vidname, "-i", project.audio, "-c:v", "libx264", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-crf",str(project.config['crf']), finalvidname])
    else:
        subprocess.call(["ffmpeg", "-i", vidname, "-c:v", "libx264", "-crf", str(project.config['crf']), finalvidname])

    #remove all temp files
    for f in [os.path.join("./temp",f) for f in os.listdir("./temp")]:
        os.remove(f) 

    print(f"TIME: {(time.time_ns()-t)/1000000}ms")
    beep()

def beep():
    winsound.Beep(440, 200)
    winsound.Beep(880, 100)


def baseButton(button, layer, project:ReactiveRGB):
    project.layers[layer].config['hasBaseLayer'] = project.layers[layer].config['hasBaseLayer'] == False
    if project.layers[layer].config['hasBaseLayer']:
        button.config(relief="sunken")
    else:
        button.config(relief="raised")

def loadUI(ui, project:ReactiveRGB):
    project.loadConfig()
    refreshUI(ui, project)

def resetUI(ui, project:ReactiveRGB):
    project.resetConfig()
    refreshUI(ui, project)
    
def refreshUI(ui, project):
    clearUI(ui)
    populateUI(ui, project)

def clearUI(ui):
    
    for thing in ui.winfo_children():
        thing.destroy()

def saveProject(project):
    # filename = askopenfilename(filetypes=[('Reactive Rainbow files','.rrgb')])
    filename= 'testthings/savefile.rrgb'
    project.save(filename)

def loadProject(ui, project):
    project.load(askopenfilename(filetypes=[('Reactive Rainbow files','.rrgb')]))
    refreshUI(ui,project)

def newLayer(ui, project):
    project.newLayer(askopenfilename())
    refreshUI(ui, project)
    
def destroyLayer(ui, project, layerID):
    project.layers.pop(layerID)
    refreshUI(ui, project)

def rainbowLayerSettings(project, k):
    layerui = tk.Tk()

    # returnButton = tk.Button(layerui, text = 'RETURN', command=refreshUI(layerui, project))
    # returnButton.grid(row =  1, column=  1)
    changeAreaButton = tk.Button(layerui, text='File', width=25, command=lambda k=k:project.layers[k].setFile(askopenfilename()))
    changeAreaButton.grid(row =  0, column=  1)
    
    isBaseButton = tk.Button(layerui, text='Is a base layer', width=25)
    isBaseButton.config(command=lambda isBaseButton=isBaseButton,k=k:baseButton(isBaseButton, k, project))
    isBaseButton.grid(row =  1, column=  1)
    if project.layers[k].config['hasBaseLayer']:
        isBaseButton.config(relief="sunken")
    else:
        isBaseButton.config(relief="raised")

    masktypeLabel = tk.Label(layerui, text='Mask type:')
    masktypeLabel.grid(row =  3, column=  1)
    pngButton = tk.Button(layerui, text='PNG', width=25, command=lambda k=k:project.layers[k].setMask('PNG'))
    maskButton = tk.Button(layerui, text='From background Mask', width=25, command=lambda k=k:project.layers[k].setMask('MASK'))
    maskFileButton = tk.Button(layerui, text='Mask', width=25, command=lambda k=k:project.layers[k].setMask(askopenfilename()))
    pngButton.grid(row =  4, column=  1)
    maskButton.grid(row =  5, column=  1)
    maskFileButton.grid(row =  6, column=  1)

    sliders = [["Rainbow Rate","Average number of seconds per rainbow rotation",project.layers[k].config["rainbowRate"],lambda val,k=k:project.layers[k].setConfig("rainbowRate",int(val)),0,100],
    ["Minimum Glow","min change area opacity",project.layers[k].config["changeAreaGlowMin"],lambda val,k=k:project.layers[k].setConfig("changeAreaGlowMin",int(val)),0,100],
    ["Maximum Glow","max change area opacity",project.layers[k].config["changeAreaGlowMax"],lambda val,k=k:project.layers[k].setConfig("changeAreaGlowMax",int(val)),0,100],
    ["Glow Radius","gaussian blur radius on glow",project.layers[k].config["changeAreaGlowRadius"],lambda val,k=k:project.layers[k].setConfig("changeAreaGlowRadius",int(val)) ,0,500],
    ["Change Area Glow","Base Change Area Glow",project.layers[k].config["changeAreaGlowBase"],lambda val,k=k:project.layers[k].setConfig("changeAreaGlowBase",int(val)),0,100],
    ["change Area linAdd","Glowy glow",project.layers[k].config["changeAreaGlowLinAdd"],lambda val,k=k:project.layers[k].setConfig("changeAreaGlowLinAdd",int(val)),0,100],
    ["max glow increase","Max rate of glow increase",project.layers[k].config["glowMaxIncrease"],lambda val,k=k:project.layers[k].setConfig("glowMaxIncrease",int(val)) ,0,100],
    ["max glow decrease","Max rate of glow decrease",project.layers[k].config["glowMaxDecrease"],lambda val,k=k:project.layers[k].setConfig("glowMaxDecrease",int(val)) ,0,100],
    ["Saturation adjust","",project.layers[k].config["saturationShift"],lambda val,k=k:project.layers[k].setConfig("saturationShift",int(val)) ,-100,100],
    ["Lightness adjust","",project.layers[k].config["luminanceShift"],lambda val,k=k:project.layers[k].setConfig("luminanceShift",int(val)) ,-100,100]]
    sliderObjects = []
    counter = 0
    for slider in sliders:
        newSlider = tk.Scale(layerui, from_=slider[4], to=slider[5], orient=tk.HORIZONTAL, command= slider[3] )
        newSlider.grid(row = counter, column = 4)
        newLabel = tk.Label(layerui, text=slider[0])
        newLabel.grid(row =  counter, column=  3)
        newDescr = tk.Label(layerui, text=slider[1])
        newDescr.grid(row =  counter, column=  5)
        newSlider.set(slider[2])
        sliderObjects.append([newSlider,newLabel,newDescr])
        counter+=1

    eqRainbowParts = []
    eqRainbowLabel = tk.Label(layerui, text="Rainbow Reactivity")
    eqRainbowLabel.grid(row =  1, column=  6)
    for n in range(len(project.EQFREQS)):
        eqRainbowParts.append({})
        eqRainbowParts[n]['slider'] = tk.Scale(layerui, from_=100, to=0, orient=tk.VERTICAL, command= lambda val,n=n:project.layers[k].setEq('eqRainbow',n, int(val)) )
        eqRainbowParts[n]['slider'].grid(row = 1, column = 7+n)
        eqRainbowParts[n]['slider'].set(project.layers[k].config["eqRainbow"][n])
        eqRainbowParts[n]['label'] = tk.Label(layerui, text=project.EQFREQS[n])
        eqRainbowParts[n]['label'].grid(row =  2, column=  7+n)

    eqGlowParts = []
    eqGlowLabel = tk.Label(layerui, text="Glow Reactivity")
    eqGlowLabel.grid(row =  3, column=  6)
    for n in range(len(project.EQFREQS)):
        eqGlowParts.append({})
        eqGlowParts[n]['slider'] = tk.Scale(layerui, from_=100, to=0, orient=tk.VERTICAL, command= lambda val,n=n:project.layers[k].setEq("eqGlow",n, int(val)) )
        eqGlowParts[n]['slider'].grid(row = 3, column = 7+n)
        eqGlowParts[n]['slider'].set(project.layers[k].config["eqGlow"][n])
        eqGlowParts[n]['label'] = tk.Label(layerui, text=project.EQFREQS[n])
        eqGlowParts[n]['label'].grid(row =  4, column=  7+n)

def populateUI(ui, project):
        #images
    # minImage = tk.Label( ui ,height= 20)
    # midImage = tk.Label(ui ,height= 20)
    # maxImage = tk.Label(ui ,height= 20)
    # minImage.grid(column = 1)
    # midImage.grid(column = 1)
    # maxImage.grid(column = 1)

    #mainbuttons
    backgroundButton = tk.Button(ui, text='Background', width=25, command=lambda:project.setBackground(askopenfilename()))

    
    audioFileButton = tk.Button(ui, text='Audio', width=25, command=lambda:project.setAudio(askopenfilename()))
    saveButton = tk.Button(ui, text="Save Settings", command=lambda:project.saveConfig())
    loadButton = tk.Button(ui, text="Load Settings", command=lambda:loadUI(ui,project))
    resetButton = tk.Button(ui, text="Reset Settings", command=lambda:resetUI(ui,project))
    setButton = tk.Button(ui, text="PREVIEW", command=lambda:preview(project))
    renderButton = tk.Button(ui, text="RENDER", command=lambda:render(project))
    # setButton = tk.Button(ui, text="SET", command=lambda:previewImage(minImage,midImage,maxImage,project))
    saveProjectButton = tk.Button(ui, text='Save Project', width=25, command=lambda:saveProject(project))
    loadProjectButton = tk.Button(ui, text='Load Project', width=25, command=lambda:loadProject(ui,project))

    buttonList = [backgroundButton,audioFileButton,saveButton,loadButton,resetButton,setButton,saveProjectButton,loadProjectButton,renderButton]
    i=0
    for num in range(len(buttonList)):
        buttonList[i].grid(row = i, column = 0)
        i+=1
        
    

    #sliders
    sliders = [
        # [label, description, start value, lambda, min, max]
        ["Frame Rate","",project.config["frameRate"],lambda val:project.setConfig("frameRate",int(val)),1,100],
        ["crf","Video Quality",project.config["crf"],lambda val:project.setConfig("crf",int(val)),1,51],
        ["thread count","",project.config["threadCount"],lambda val:project.setConfig("threadCount",int(val)),1,64],
        ["max RAM (GB)","THIS IS AN ESTIMATE",project.config["maxRAM"],lambda val:project.setConfig("maxRAM",int(val)),1,64],
        ["db floor","Percentage of values considered '0'",project.config["dbPercentileFloor"],lambda val:project.setConfig("dbPercentileFloor",int(val)) ,0,100],
        ["db ceiling","Percentage of values considered '100'",project.config["dbPercentileCeiling"],lambda val:project.setConfig("dbPercentileCeiling",int(val)),0,100],
        ["Boom MAX","Maximum amount image can grow",project.config["maxBoom"],lambda val:project.setConfig("maxBoom",int(val)),0,100],
        ["boomoffset","shift boom by frames",project.config["boomoffset"],lambda val:project.setConfig("boomoffset",int(val)),-50,50],
        ["boomwinlen","softening range for boom",project.config["boomwinlen"],lambda val:project.setConfig("boomwinlen",int(val)),1,100],
        ["boompolyorder","",project.config["boompolyorder"],lambda val:project.setConfig("boompolyorder",int(val)),1,10]        
    ]
    
    sliderObjects = []
    counter = 0
    for slider in sliders:
        newSlider = tk.Scale(ui, from_=slider[4], to=slider[5], orient=tk.HORIZONTAL, command= slider[3] )
        newSlider.grid(row = counter, column = 4)
        newLabel = tk.Label(ui, text=slider[0])
        newLabel.grid(row =  counter, column=  3)
        newDescr = tk.Label(ui, text=slider[1])
        newDescr.grid(row =  counter, column=  5)
        newSlider.set(slider[2])
        sliderObjects.append([newSlider,newLabel,newDescr])
        counter+=1


    eqBoomParts = []
    eqBoomLabel = tk.Label(ui, text="BOOM Reactivity")
    eqBoomLabel.grid(row =  1, column=  6)
    for n in range(len(project.EQFREQS)):
        eqBoomParts.append({})
        eqBoomParts[n]['slider'] = tk.Scale(ui, from_=100, to=0, orient=tk.VERTICAL, command= lambda val,n=n:project.setEqBoom(n, int(val)) )
        eqBoomParts[n]['slider'].grid(row = 1, column = 7+n)
        eqBoomParts[n]['slider'].set(project.config["eqBoom"][n])
        eqBoomParts[n]['label'] = tk.Label(ui, text=project.EQFREQS[n])
        eqBoomParts[n]['label'].grid(row =  2, column=  7+n)

    
    layerbuttons = []
    i=0
    for k in project.layers.keys():
        layerbuttons.append([])
        layerbuttons[i].append(tk.Label(ui, text=project.layers[k].imgFile))
        layerbuttons[i].append(tk.Button(ui, text="EDIT", command=lambda k=k:rainbowLayerSettings(project, k)))
        layerbuttons[i].append(tk.Button(ui, text="DELETE", command=lambda k=k:destroyLayer(ui, project,k)))
        layerbuttons[i][0].grid(row = i+4, column = 7)
        layerbuttons[i][1].grid(row = i+4, column = 8)
        layerbuttons[i][2].grid(row = i+4, column = 9)
        i+=1
    addLayerButton = tk.Button(ui, text='add layer', command=lambda:newLayer(ui, project))
    addLayerButton.grid(row = i+4, column = 7)

def UI(project:ReactiveRGB = None):
    if project is None: project = ReactiveRGB()
    ui = tk.Tk()
    ui.title('Rainbowing Audio')
    populateUI(ui, project)
    


    ui.mainloop()
if __name__ == "__main__":
    UI()