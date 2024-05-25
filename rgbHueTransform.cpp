#include <math.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

extern "C"
{
    typedef unsigned char BYTE; // define an "integer" that only stores 0-255 value

    // typedef struct _CRGB // Define a struct to store the 3 color values
    // {
    //     BYTE r;
    //     BYTE g;
    //     BYTE b;
    // } CRGB;

    struct tMatrix
    {
        float rr;
        float rg;
        float rb;
        float gr;
        float gg;
        float gb;
        float br;
        float bg;
        float bb;
    };

    BYTE clamp(float v) // define a function to bound and round the input float value to 0-255
    {
        if (v < 0)
            return 0;
        if (v > 255)
            return 255;
        return (BYTE)v;
    }
    tMatrix prepMatrix(float hue)
    {
        tMatrix m;
        const float cosA = cos(hue * 3.14159265f / 180);
        const float sinA = sin(hue * 3.14159265f / 180);
        m.rr = cosA + (1.0f - cosA) / 3.0f;
        m.rg = 1.0f / 3.0f * (1.0f - cosA) - sqrtf(1.0f / 3.0f) * sinA;
        m.rb = 1.0f / 3.0f * (1.0f - cosA) + sqrtf(1.0f / 3.0f) * sinA;
        m.gr = 1.0f / 3.0f * (1.0f - cosA) + sqrtf(1.0f / 3.0f) * sinA;
        m.gg = cosA + 1.0f / 3.0f * (1.0f - cosA);
        m.gb = 1.0f / 3.0f * (1.0f - cosA) - sqrtf(1.0f / 3.0f) * sinA;
        m.br = 1.0f / 3.0f * (1.0f - cosA) - sqrtf(1.0f / 3.0f) * sinA;
        m.bg = 1.0f / 3.0f * (1.0f - cosA) + sqrtf(1.0f / 3.0f) * sinA;
        m.bb = cosA + 1.0f / 3.0f * (1.0f - cosA);
        return m;
    }
    float LinearToGamma(float value, float gamma)
    {
        return pow(value, 1.0 / gamma) * 255;
    }

    float GammaToLinear(float value, float gamma)
    {
        return pow(value / 255.0, gamma);
    }

    // HSV version
    API void TransformImageHSV(int *img, int hue, int pixelcount, bool transparent = false, float gamma = 2.2)
    {
        // if (hue==0){return;}

        int incr = 3;
        if (transparent)
        {
            int incr = 4;
        }

        tMatrix m = prepMatrix(hue);
        pixelcount = pixelcount * incr;
        for (int x = 0; x < pixelcount; x = x + incr)
        {
            // float r = GammaToLinear(img[x], gamma);
            // float g = GammaToLinear(img[x + 1], gamma);
            // float b = GammaToLinear(img[x + 2], gamma);
            float r = img[x];
            float g = img[x + 1];
            float b = img[x + 2];
            r = clamp(r * m.rr + g * m.rg + b * m.rb);
            g = clamp(r * m.gr + g * m.gg + b * m.gb);
            b = clamp(r * m.br + g * m.bg + b * m.bb);
            // img[x] = LinearToGamma(r, gamma);
            // img[x + 1] = LinearToGamma(g, gamma);
            // img[x + 2] = LinearToGamma(b, gamma);
            img[x] = (int)r;
            img[x + 1] = (int)g;
            img[x + 2] = (int)b;
        }
    }

    float tempColourCalc(float temp1, float temp2, float tempC)
    {
        if (tempC < 0.1666667)
        {
            return temp2 + ((temp1 - temp2) * 6 * tempC);
        }
        else if (tempC < 0.5)
        {
            return temp1;
        }
        else if (tempC < 0.6666667)
        {
            return temp2 + ((temp1 - temp2) * (0.6666667 - tempC) * 6);
        }
        else
        {
            return temp2;
        }
    }

    // HSL version
    API void TransformImageHSL(int *img, int pixelcount, float hueShift, float saturationShift = 0, float luminanceShift = 0, int incr = 3)
    {

        tMatrix m = prepMatrix(hueShift);
        pixelcount = pixelcount * incr;

        float h = 0;
        float s = 0;
        float l = 0;
        float rf = 0;
        float gf = 0;
        float bf = 0;
        float maxf = 0;
        float minf = 0;
        float temp1;
        float temp2;
        float tempR;
        float tempG;
        float tempB;
        for (int x = 0; x < pixelcount; x = x + incr)
        {
            float rf = img[x] / 255.0;
            float gf = img[x + 1] / 255.0;
            float bf = img[x + 2] / 255.0;

            if (rf <= gf)
            {
                minf = rf;
                maxf = gf;
            }
            else
            {
                minf = gf;
                maxf = rf;
            }
            if (bf < minf)
            {
                minf = bf;
            }
            else if (bf > maxf)
            {
                maxf = bf;
            }

            // luminance calculation
            l = (minf + maxf) / 2;

            // saturation calculation
            if (minf == maxf)
            {
                s = 0;
            }
            else if (l > 0.5)
            {
                s = (maxf - minf) / (2.0 - maxf - minf);
            }
            else
            {
                s = (maxf - minf) / (maxf + minf);
            }
            // hue calculation
            if (minf == maxf)
            {
                h = 0;
            }
            else if (maxf == rf)
            { // RED
                h = (gf - bf) / (maxf - minf);
            }
            else if (maxf == gf)
            { // GREEN
                h = 2.0 + (bf - rf) / (maxf - minf);
            }
            else
            { // BLUE
                h = 4.0 + (rf - gf) / (maxf - minf);
            }
            h = h * 60;

            h = h + hueShift;
            s = s + saturationShift;
            l = l + luminanceShift;
            if (h > 360)
            {
                h = h - 360;
            }
            else if (h < 0)
            {
                h = h + 360;
            }
            if (s > 1)
            {
                s = 1;
            }
            if (l > 1)
            {
                l = 1;
            }

            // convert back
            if (l < 0.5)
            {
                temp1 = l * (1 + s);
            }
            else
            {
                temp1 = l + s - (l * s);
            }
            temp2 = 2 * l - temp1;

            h = h / 360;
            tempR = h + 0.333;
            if (tempR > 1)
            {
                tempR = tempR - 1;
            }
            tempG = h;
            tempB = h - 0.333;
            if (tempB < 0)
            {
                tempB = tempB + 1;
            }

            img[x] = tempColourCalc(temp1, temp2, tempR) * 255;
            img[x + 1] = tempColourCalc(temp1, temp2, tempG) * 255;
            img[x + 2] = tempColourCalc(temp1, temp2, tempB) * 255;
        }
    }

    API void LinearAdd(int *img1, int *img2, int pixelCount, float alpha, bool transparent1 = false, bool transparent2 = false)
    {
        if (alpha > 1)
        {
            alpha = 1;
        }
        else if (alpha < 0)
        {
            alpha = 0;
        }
        int img1multiple = 3;
        if (transparent1)
        {
            img1multiple = 4;
        }
        int img2multiple = 3;
        if (transparent2)
        {
            img2multiple = 4;
        }
        int x = 0;
        while (x < pixelCount)
        {
            if (transparent2)
            {
                img1[x * img1multiple] = img1[x * img1multiple] + img2[x * img2multiple] * alpha * img2[x * img2multiple + 3] / 255;
                img1[x * img1multiple + 1] = img1[x * img1multiple + 1] + img2[x * img2multiple + 1] * alpha * img2[x * img2multiple + 3] / 255;
                img1[x * img1multiple + 2] = img1[x * img1multiple + 2] + img2[x * img2multiple + 2] * alpha * img2[x * img2multiple + 3] / 255;
            }
            else
            {
                img1[x * img1multiple] = img1[x * img1multiple] + img2[x * img2multiple] * alpha;
                img1[x * img1multiple + 1] = img1[x * img1multiple + 1] + img2[x * img2multiple + 1] * alpha;
                img1[x * img1multiple + 2] = img1[x * img1multiple + 2] + img2[x * img2multiple + 2] * alpha;
            }
            if (img1[x * img1multiple] > 255)
            {
                img1[x * img1multiple] = 255;
            }
            if (img1[x * img1multiple + 1] > 255)
            {
                img1[x * img1multiple + 1] = 255;
            }
            if (img1[x * img1multiple + 2] > 255)
            {
                img1[x * img1multiple + 2] = 255;
            }
            x++;
        }
    }

    API void AudioFormatter(float *output, float *input, float *freq, float *time, int framecount, int timeCount, int freqCount, int framerate)
    {
        int freqEnds[10] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 24576};
        int freqPlace = 0;
        int counter = 0;

        int t = 0;
        for (int frame = 0; frame < framecount; frame++) //Iterate through the frames of the output array
        {
            while(time[t]*framerate<frame){ //Sync the time of the time array to the frame array
                t++;
            }
            for (int f = 0; f < freqCount; f++) //iterates through the frequency of the input array
            {
                if(freq[f]<freqEnds[freqPlace]){ //iterates through the frequency of the output array
                    counter++;
                }else{
                    output[frame*10 + freqPlace] = output[frame*10 + freqPlace]/counter;
                    counter=0;
                    freqPlace++;
                }
                output[frame*10 + freqPlace] =  output[frame*10 +freqPlace] + input[t*freqCount+f];
            }
            output[frame*10 + freqPlace] = output[frame*10 + freqPlace]/counter;
            counter=0;
            freqPlace = 0;
        }
    }
}

int main()
{
}
