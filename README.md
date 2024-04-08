# CUDA Median Filter
Provide a median filter sample based on GPU acceleration(cuda), and compare it to traditional opencv 
cv::medianFilter in several case;

achieve code has been provide in  [main.cu](main.cu)

### float 
cv::medianFiler only provide filer_size=3、5 VERSION in CV_32FC1     
test count=999;     

Due to the GPU needing to warm up during the first run, exclude the first result        
the first result of gpu nearly 100ms;       

filter_size = 3、5、7     
Data transmission time in parentheses

| Size      | cpu (3) | gpu (3)      | cpu (5) | gpu (5)     |
|-----------|---------|--------------|---------|-------------|
| 256x256   | 0.0647  | 0.139(0.077) |         |             |
| 512*512   | 0.2020  | 0.346(0.198) |         |             |         
| 1024*1024 | 0.6651  | 0.554(0.961) |         |             |
| 2048*2048 | 2.4062  | 1.855(3.364) |         |             |
| 4000*3000 | 6.5132  | 4.782(8.940) |         |             |
| 5120*5120 | 14.533  | 10.05(19.9)  | 96.0208 | 47.63(19.9) |
spand time / ms

This is a reference.It can be optimized.        
It better be used if you have a cuda Project.

