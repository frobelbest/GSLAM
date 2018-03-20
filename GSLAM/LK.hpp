//
//  trackFeatures.hpp
//  STARCK
//
//  Created by Chaos on 4/1/16.
//  Copyright © 2016 Chaos. All rights reserved.
//

#include "KLT.h"
#include "ImageGrids.hpp"
#include "error.h"
#include <stdlib.h>
#include <avxintrin.h>
#include "tbb/tbb.h"
#include "Homography.hpp"
#include <utility>
#include "opencv2/imgproc/imgproc.hpp"

const bool compensate_motion=false;
const bool compensate_lighting=false;

static float lighting_alpha;
static float lighting_beta;
cv::Mat      homography;


int KLTCountRemainingFeatures(KLT_FeatureList fl){
    
    int count = 0;
    int i;
    
    for (i = 0 ; i < fl->nFeatures ; i++){
        if (fl->feature[i]->val >= 0){
            count++;
        }
    }
    return count;
}

const int simd_step=8;


static void compensatePatchLighting(float* patch,
                                    const int width,const int height,
                                    const int alignedPatchSize,const float beta){
    
    __m256 _beta = _mm256_set1_ps(beta);
    for(int p=0;p<width*height;p+=simd_step){
        __m256 _patch=_mm256_add_ps(_mm256_load_ps(&patch[p]),_beta);
        _mm256_store_ps(&patch[p],_patch);
    }
    memset(&patch[width*height],0,(alignedPatchSize-width*height)*sizeof(float));
    return;
}

static void computeBilinearWeight(float x,float y,int &xt,int &yt,float weights[4]){
    
    xt=(int)x;
    yt=(int)y;
    
    float ax=x-xt;
    float ay=y-yt;
    
    weights[0]=(1.0-ax)*(1.0-ay);
    weights[1]=(ax)*(1.0-ay);
    weights[2]=(1.0-ax)*ay;
    weights[3]=ax*ay;
}


static void computeBilinearPatch(float* patch,
                                 const cv::Mat &image,
                                 const int xt,const int yt,
                                 const int width,const int height,
                                 const int alignedPatchSize,
                                 const float weights[4]){
    
    int hw = width/2, hh = height/2;
    
    if(weights[0]==1.0){
        for (int h=-hh,y=0;h<=hh;h++,y++) {
            memcpy(&patch[y*width],&image.at<float>(yt+h,xt-hw),width*sizeof(float));
        }
        return;
    }
    
    __m256 weight00 = _mm256_set1_ps(weights[0]);
    __m256 weight01 = _mm256_set1_ps(weights[1]);
    __m256 weight10 = _mm256_set1_ps(weights[2]);
    __m256 weight11 = _mm256_set1_ps(weights[3]);
    
    for (int h=-hh,y=0;h<=hh;h++,y++) {
        float *imagePtr0=(float*)image.ptr(h+yt);
        float *imagePtr1=(float*)image.ptr(h+yt+1);
        
        for (int w=-hw,x=0;w<=hw;w+=simd_step,x+=simd_step) {
            
            __m256 patch00=_mm256_mul_ps(_mm256_load_ps(&imagePtr0[xt+w]),weight00);
            __m256 patch01=_mm256_mul_ps(_mm256_load_ps(&imagePtr0[xt+w+1]),weight01);
            __m256 patch10=_mm256_mul_ps(_mm256_load_ps(&imagePtr1[xt+w]),weight10);
            __m256 patch11=_mm256_mul_ps(_mm256_load_ps(&imagePtr1[xt+w+1]),weight11);
            
            __m256 newpatch=_mm256_add_ps(_mm256_add_ps(patch00,patch01),_mm256_add_ps(patch10,patch11));
            _mm256_store_ps(&patch[y*width+x],newpatch);
        }
    }
    memset(&patch[width*height],0,(alignedPatchSize-width*height)*sizeof(float));
}

static void computeImageDiff(const float* patch1,const float* patch2,float* diff,const int patchsize){
    for (int i=0;i<patchsize;i+=simd_step) {
        _mm256_store_ps(&diff[i],_mm256_sub_ps(_mm256_load_ps(&patch1[i]),
                                               _mm256_load_ps(&patch2[i])));
        
    }
}

static void computeImageSum(const float* patch1,const float* patch2,float* sum,const int patchsize){
    
    for (int i=0;i<patchsize;i+=simd_step) {
        _mm256_store_ps(&sum[i],_mm256_add_ps(_mm256_load_ps(&patch1[i]),
                                              _mm256_load_ps(&patch2[i])));
        
    }
}

static void compute2X2GradientMatrix(const float* gradx,const float* grady,int patchsize,float &gxx,float &gyy,float &gxy){
    
    float gxx_buf[simd_step],gyy_buf[simd_step],gxy_buf[simd_step];
    
    __m256 gxx_reg=_mm256_setzero_ps(),gyy_reg=_mm256_setzero_ps(),gxy_reg=_mm256_setzero_ps();

    for (int i=0;i<patchsize;i+=simd_step) {
        gxx_reg=_mm256_add_ps(gxx_reg,_mm256_mul_ps(_mm256_load_ps(&gradx[i]),_mm256_load_ps(&gradx[i])));
        gyy_reg=_mm256_add_ps(gyy_reg,_mm256_mul_ps(_mm256_load_ps(&grady[i]),_mm256_load_ps(&grady[i])));
        gxy_reg=_mm256_add_ps(gxy_reg,_mm256_mul_ps(_mm256_load_ps(&gradx[i]),_mm256_load_ps(&grady[i])));
        
    }
    _mm256_store_ps(gxx_buf,gxx_reg);
    _mm256_store_ps(gyy_buf,gyy_reg);
    _mm256_store_ps(gxy_buf,gxy_reg);
    
    gxx=gxx_buf[0]+gxx_buf[1]+gxx_buf[2]+gxx_buf[3]+gxx_buf[4]+gxx_buf[5]+gxx_buf[6]+gxx_buf[7];
    gyy=gyy_buf[0]+gyy_buf[1]+gyy_buf[2]+gyy_buf[3]+gyy_buf[4]+gyy_buf[5]+gyy_buf[6]+gyy_buf[7];
    gxy=gxy_buf[0]+gxy_buf[1]+gxy_buf[2]+gxy_buf[3]+gxy_buf[4]+gxy_buf[5]+gxy_buf[6]+gxy_buf[7];
}

static void compute2X1ErrorVector(const float* diff,const float* gradx,const float* grady,int patchsize,float step_factor,float &ex,float &ey){
    
    float ex_buf[simd_step],ey_buf[simd_step];

    __m256 ex_reg=_mm256_setzero_ps(),ey_reg=_mm256_setzero_ps();
    
    for (int i=0;i<patchsize;i+=simd_step) {
        ex_reg=_mm256_add_ps(ex_reg,_mm256_mul_ps(_mm256_load_ps(&gradx[i]),_mm256_load_ps(&diff[i])));
        ey_reg=_mm256_add_ps(ey_reg,_mm256_mul_ps(_mm256_load_ps(&grady[i]),_mm256_load_ps(&diff[i])));
    }
    
    _mm256_store_ps(ex_buf,ex_reg);
    _mm256_store_ps(ey_buf,ey_reg);

    
    ex=(ex_buf[0]+ex_buf[1]+ex_buf[2]+ex_buf[3]+ex_buf[4]+ex_buf[5]+ex_buf[6]+ex_buf[7])*step_factor;
    ey=(ey_buf[0]+ey_buf[1]+ey_buf[2]+ey_buf[3]+ey_buf[4]+ey_buf[5]+ey_buf[6]+ey_buf[7])*step_factor;
    
}

static int _solveEquation(float gxx, float gxy, float gyy,
                          float ex, float ey,
                          float small,
                          float *dx, float *dy)
{
    float det = gxx*gyy - gxy*gxy;
    
    if (det < small){
        
        return KLT_SMALL_DET;
    }
    
    *dx = (gyy*ex - gxy*ey)/det;
    *dy = (gxx*ey - gxy*ex)/det;
    return KLT_TRACKED;
}

static bool _outOfBounds(float x,float y,
                         int ncols,int nrows,
                         int borderx,int bordery)
{
    return (x < borderx || x > ncols-1-borderx ||
            y < bordery || y > nrows-1-bordery );
}

static float computeABSImageDiff(float *patchdiff,const int patchsize){
    
    __m256 a = _mm256_set1_ps(-0.0);
    __m256 absSum=_mm256_setzero_ps();
    
    for (int i=0;i<patchsize;i+=simd_step) {
        __m256 value = _mm256_load_ps(&patchdiff[i]);
        value=_mm256_andnot_ps(a,value);
        absSum=_mm256_add_ps(absSum,value);
    }
    
    float sumBuffer[simd_step];
    _mm256_store_ps(sumBuffer,absSum);
    
    return sumBuffer[0]+sumBuffer[1]+sumBuffer[2]+sumBuffer[3]+sumBuffer[4]+sumBuffer[5]+sumBuffer[6]+sumBuffer[7];
}


typedef float *_FloatWindow;

typedef struct  {
    int ncols;
    int nrows;
    float *data;
}  _KLT_FloatImageRec, *_KLT_FloatImage;

typedef struct  {
    int subsampling;
    int nLevels;
    _KLT_FloatImage *img;
    int *ncols, *nrows;
}  _KLT_PyramidRec, *_KLT_Pyramid;


static float _interpolate(float x,
                          float y,
                          _KLT_FloatImage img)
{
    int xt = (int) x;  /* coordinates of top-left corner */
    int yt = (int) y;
    float ax = x - xt;
    float ay = y - yt;
    float *ptr = img->data + (img->ncols*yt) + xt;

    assert (xt >= 0 && yt >= 0 && xt <= img->ncols - 2 && yt <= img->nrows - 2);
    //printf("coef %f %f\n",ax,ay);
    return ( (1-ax) * (1-ay) * *ptr +
            ax   * (1-ay) * *(ptr+1) +
            (1-ax) *   ay   * *(ptr+(img->ncols)) +
            ax   *   ay   * *(ptr+(img->ncols)+1) );
}


/*********************************************************************
 * _computeIntensityDifference
 *
 * Given two images and the window center in both images,
 * aligns the images wrt the window and computes the difference
 * between the two overlaid images.
 */

static void _computeIntensityDifference(
                                        _KLT_FloatImage img1,   /* images */
                                        _KLT_FloatImage img2,
                                        float x1, float y1,     /* center of window in 1st img */
                                        float x2, float y2,     /* center of window in 2nd img */
                                        int width, int height,  /* size of window */
                                        _FloatWindow imgdiff)   /* output */
{
    
    register int hw = width/2, hh = height/2;
    float g1, g2;
    register int i, j;
    
    /* Compute values */
    for (j = -hh ; j <= hh ; j++)
        for (i = -hw ; i <= hw ; i++)  {
            g1 = _interpolate(x1+i, y1+j, img1);
            g2 = _interpolate(x2+i, y2+j, img2);
            *imgdiff++ = g1 - g2;
        }
}


/*********************************************************************
 * _computeGradientSum
 *
 * Given two gradients and the window center in both images,
 * aligns the gradients wrt the window and computes the sum of the two
 * overlaid gradients.
 */

static void _computeGradientSum(
                                _KLT_FloatImage gradx1,  /* gradient images */
                                _KLT_FloatImage grady1,
                                _KLT_FloatImage gradx2,
                                _KLT_FloatImage grady2,
                                float x1, float y1,      /* center of window in 1st img */
                                float x2, float y2,      /* center of window in 2nd img */
                                int width, int height,   /* size of window */
                                _FloatWindow gradx,      /* output */
                                _FloatWindow grady)      /*   " */
{
    register int hw = width/2, hh = height/2;
    float g1, g2;
    register int i, j;
    
    /* Compute values */
    for (j = -hh ; j <= hh ; j++)
        for (i = -hw ; i <= hw ; i++)  {
            g1 = _interpolate(x1+i, y1+j, gradx1);
            g2 = _interpolate(x2+i, y2+j, gradx2);
            *gradx++ = g1 + g2;
            g1 = _interpolate(x1+i, y1+j, grady1);
            g2 = _interpolate(x2+i, y2+j, grady2);
            *grady++ = g1 + g2;
        }
}


/*********************************************************************
 * _compute2by2GradientMatrix
 *
 */

static void _compute2by2GradientMatrix(
                                       _FloatWindow gradx,
                                       _FloatWindow grady,
                                       int width,   /* size of window */
                                       int height,
                                       float *gxx,  /* return values */
                                       float *gxy,
                                       float *gyy)

{
    register float gx, gy;
    register int i;
    
    /* Compute values */
    *gxx = 0.0;  *gxy = 0.0;  *gyy = 0.0;
    for (i = 0 ; i < width * height ; i++)  {
        gx = *gradx++;
        gy = *grady++;
        *gxx += gx*gx;
        *gxy += gx*gy;
        *gyy += gy*gy;
    }
}


/*********************************************************************
 * _compute2by1ErrorVector
 *
 */

static void _compute2by1ErrorVector(
                                    _FloatWindow imgdiff,
                                    _FloatWindow gradx,
                                    _FloatWindow grady,
                                    int width,   /* size of window */
                                    int height,
                                    float step_factor, /* 2.0 comes from equations, 1.0 seems to avoid overshooting */
                                    float *ex,   /* return values */
                                    float *ey)
{
    register float diff;
    register int i;
    
    /* Compute values */
    *ex = 0;  *ey = 0;  
    for (i = 0 ; i < width * height ; i++)  {
        diff = *imgdiff++;
        *ex += diff * (*gradx++);
        *ey += diff * (*grady++);
    }
    *ex *= step_factor;
    *ey *= step_factor;
}




/*********************************************************************
 * _allocateFloatWindow
 */

static _FloatWindow _allocateFloatWindow(
                                         int width,
                                         int height)
{
    _FloatWindow fw;
    
    fw = (_FloatWindow) malloc(width*height*sizeof(float));
    if (fw == NULL)  KLTError("(_allocateFloatWindow) Out of memory.");
    return fw;
}



static float _sumAbsFloatWindow(
                                _FloatWindow fw,
                                int width,
                                int height)
{
    float sum = 0.0;
    int w;
    
    for ( ; height > 0 ; height--)
        for (w=0 ; w < width ; w++)
            sum += (float) fabs(*fw++);
    
    return sum;
}







static int _trackFeature(float x1,  /* location of window in first image */
                         float y1,
                         float *x2, /* starting location of search in second image */
                         float *y2,
                         cv::Mat &img1,
                         cv::Mat &gradx1,
                         cv::Mat &grady1,
                         cv::Mat &img2,
                         cv::Mat &gradx2,
                         cv::Mat &grady2,
                         int width,           /* size of window */
                         int height,
                         float step_factor, /* 2.0 comes from equations, 1.0 seems to avoid overshooting */
                         int max_iterations,
                         float small,         /* determinant threshold for declaring KLT_SMALL_DET */
                         float th,            /* displacement threshold for stopping               */
                         float max_residue,   /* residue threshold for declaring KLT_LARGE_RESIDUE */
                         int   lighting_insensitive)  /* whether to normalize for gain and bias */
{
    
    float gxx, gxy, gyy, ex, ey, dx, dy;
    int iteration = 0;
    int status;
    int hw = width/2;
    int hh = height/2;
    int nc = img1.cols;
    int nr = img1.rows;
    float one_plus_eps = 1.001f;   /* To prevent rounding errors */
    
    float weights1[4],weights2[4];
    
    
    
#ifdef SSE_TRACKING
    
    int alignedPatchSize=simd_step*((width*height+simd_step)/simd_step);
    
    
    float *buffers=(float*)malloc(9*alignedPatchSize*sizeof(float));
    memset(buffers,0,9*alignedPatchSize*sizeof(float));
    
    float *patch1=buffers;
    float *gx1=&patch1[alignedPatchSize];
    float *gy1=&gx1[alignedPatchSize];
    
    float *patch2=&gy1[alignedPatchSize];
    float *gx2=&patch2[alignedPatchSize];
    float *gy2=&gx2[alignedPatchSize];
    
    float *patchdiff=&gy2[alignedPatchSize];
    float *gx=&patchdiff[alignedPatchSize];
    float *gy=&gx[alignedPatchSize];
    

    int xt1,yt1;
    computeBilinearWeight(x1,y1,xt1,yt1,weights1);
    
    if(compensate_lighting){
        for(int i=0;i<4;i++)
            weights1[i]*=lighting_alpha;
    }
    computeBilinearPatch(patch1,img1,xt1,yt1,width,height,alignedPatchSize,weights1);
    computeBilinearPatch(gx1,gradx1,xt1,yt1,width,height,alignedPatchSize,weights1);
    computeBilinearPatch(gy1,grady1,xt1,yt1,width,height,alignedPatchSize,weights1);
    
    if(compensate_lighting){
        //printf("aligned size %d\n",alignedPatchSize);
        /*float _patch[alignedPatchSize];
        memset(&_patch[width*height],0,(alignedPatchSize-width*height)*sizeof(float));
        for(int i=0;i<width*height;i++){
            _patch[i]=patch1[i]+lighting_beta;
        }*/
        compensatePatchLighting(patch1,width,height,alignedPatchSize,lighting_beta);
        
        
        /*for(int i=0;i<alignedPatchSize;i++){
            //printf("%f %f\n",patch1[i],_patch[i]);
            //assert(patch1[i]==_patch[i]);
            if(patch1[i]!=_patch[i]){
                printf("not equal %d %f %f %f\n",i,patch1[i],_patch[i],lighting_beta);
                assert(0);
            }
        }*/
        
        /*for(int i=0;i<alignedPatchSize;i++){
            //printf("%f %f\n",patch1[i],_patch[i]);
            if(patch1[i]!=_patch[i]){
                printf("%d %f %f %f\n",i,patch1[i],_patch[i],lighting_beta);
                assert(0);
            }
            //assert(patch1[i]==_patch[i]);
        }
        getchar();*/
    }
    
#else
    _FloatWindow imgdiff = _allocateFloatWindow(width, height);
    _FloatWindow gradx   = _allocateFloatWindow(width, height);
    _FloatWindow grady   = _allocateFloatWindow(width, height);
    
    _KLT_FloatImage kltimage1,kltimage2,kltgradx1,kltgradx2,kltgrady1,kltgrady2;
    
    kltimage1=(_KLT_FloatImage)malloc(sizeof(_KLT_FloatImageRec));
    kltimage2=(_KLT_FloatImage)malloc(sizeof(_KLT_FloatImageRec));
    
    kltgradx1=(_KLT_FloatImage)malloc(sizeof(_KLT_FloatImageRec));
    kltgradx2=(_KLT_FloatImage)malloc(sizeof(_KLT_FloatImageRec));
    
    kltgrady1=(_KLT_FloatImage)malloc(sizeof(_KLT_FloatImageRec));
    kltgrady2=(_KLT_FloatImage)malloc(sizeof(_KLT_FloatImageRec));
    
    
    kltimage1->data=(float*)img1.data;
    kltimage1->ncols=img1.cols;
    kltimage1->nrows=img1.rows;
    
    kltimage2->data=(float*)img2.data;
    kltimage2->ncols=img2.cols;
    kltimage2->nrows=img2.rows;
    
    kltgradx1->data=(float*)gradx1.data;
    kltgradx1->ncols=gradx1.cols;
    kltgradx1->nrows=gradx1.rows;
    
    
    kltgradx2->data=(float*)gradx2.data;
    kltgradx2->ncols=gradx2.cols;
    kltgradx2->nrows=gradx2.rows;
    
    kltgrady1->data=(float*)grady1.data;
    kltgrady1->ncols=grady1.cols;
    kltgrady1->nrows=grady1.rows;

    
    kltgrady2->data=(float*)grady2.data;
    kltgrady2->ncols=grady2.cols;
    kltgrady2->nrows=grady2.rows;
#endif
    

    //printf("%f %f\n",x1,y1);
    /* Iteratively update the window position */
    do  {
        
        /* If out of bounds, exit loop */
        if ( x1-hw < 0.0f || nc-( x1+hw) < one_plus_eps ||
            *x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps ||
             y1-hh < 0.0f || nr-( y1+hh) < one_plus_eps ||
            *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps) {
            status = KLT_OOB;
            break;
        }
        
#ifdef SSE_TRACKING
        
        int xt2,yt2;
        computeBilinearWeight(*x2,*y2,xt2,yt2,weights2);
        computeBilinearPatch(patch2,img2,xt2,yt2,width,height,alignedPatchSize,weights2);
        computeBilinearPatch(gx2,gradx2,xt2,yt2,width,height,alignedPatchSize,weights2);
        computeBilinearPatch(gy2,grady2,xt2,yt2,width,height,alignedPatchSize,weights2);

        
        computeImageDiff(patch1,patch2,patchdiff,width*height);
        computeImageSum(gx1,gx2,gx,width*height);
        computeImageSum(gy1,gy2,gy,width*height);
        

        compute2X2GradientMatrix(gx,gy,width*height,gxx,gyy,gxy);
        compute2X1ErrorVector(patchdiff,gx,gy,alignedPatchSize,step_factor,ex,ey);
#else
        
        
        _computeIntensityDifference(kltimage1,kltimage2, x1, y1, *x2, *y2,
                                    width, height, imgdiff);
        
        
        
        
        
        _computeGradientSum(kltgradx1,kltgrady1,kltgradx2,kltgrady2,
                            x1, y1, *x2, *y2, width, height, gradx, grady);
        
        
        _compute2by2GradientMatrix(gradx, grady, width, height,
                                   &gxx, &gxy, &gyy);
        
        _compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor,
                                &ex, &ey);
#endif
        
        /* Using matrices, solve equation for new displacement */
        status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
        
        if (status == KLT_SMALL_DET){
            break;
        }
        
        *x2 += dx;
        *y2 += dy;
        iteration++;
        //printf("%f %f\n",dx,dy);
        
    }  while ((fabs(dx)>=th || fabs(dy)>=th) && iteration < max_iterations);
    //printf("max iter %d\n",max_iterations);
    //getchar();
    
    /* Check whether window is out of bounds */
    if (*x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps ||
        *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps){
        status = KLT_OOB;
    }
    
    /* Check whether residue is too large */
    if (status == KLT_TRACKED)  {
        
#ifdef SSE_TRACKING
        int xt2,yt2;
        computeBilinearWeight(*x2,*y2,xt2,yt2,weights2);
        computeBilinearPatch(patch2,img2,xt2,yt2,width,height,alignedPatchSize,weights2);
        computeImageDiff(patch1,patch2,patchdiff,alignedPatchSize);
        
        if (computeABSImageDiff(patchdiff,alignedPatchSize)/(width*height)> max_residue){
            status = KLT_LARGE_RESIDUE;
        }
#else
        
        _computeIntensityDifference(kltimage1,kltimage2, x1, y1, *x2, *y2,
                                    width, height, imgdiff);
        float diff2=_sumAbsFloatWindow(imgdiff, width, height)/(width*height);
        if (_sumAbsFloatWindow(imgdiff, width, height)/(width*height) > max_residue)
            status = KLT_LARGE_RESIDUE;
#endif
        
    }
#ifdef SSE_TRACKING
    free(buffers);
#else
    free(imgdiff);  free(gradx);  free(grady);
#endif
    
    
    /* Return appropriate value */
    if (status == KLT_SMALL_DET){
        return KLT_SMALL_DET;
    }
    else if (status == KLT_OOB){
        return KLT_OOB;
    }
    else if (status == KLT_LARGE_RESIDUE){
        return KLT_LARGE_RESIDUE;
    }
    else if (iteration >= max_iterations) {
        return KLT_MAX_ITERATIONS;
    }
    else  return KLT_TRACKED;
    
}

static int refineFeatureTranslation(float *x2,
                                    float *y2,
                                    cv::Mat img1,
                                    cv::Mat gradx1,
                                    cv::Mat grady1,
                                    cv::Mat img2,
                                    cv::Mat gradx2,
                                    cv::Mat grady2,
                                    int width,           /* size of window */
                                    int height,
                                    float step_factor, /* 2.0 comes from equations, 1.0 seems to avoid overshooting */
                                    int max_iterations,
                                    float small,         /* determinant threshold for declaring KLT_SMALL_DET */
                                    float th,
                                    float max_residue,   /* residue threshold for declaring KLT_LARGE_RESIDUE */
                                    float mdd)        /* used affine mapping */
{
    
    float gxx, gxy, gyy, ex, ey, dx, dy;
    int iteration = 0;
    int status = 0;
    int hw = width/2;
    int hh = height/2;
    
    int nc2 = img2.cols;
    int nr2 = img2.rows;
    

    float one_plus_eps = 1.001f;   /* To prevent rounding errors */
    float old_x2 = *x2;
    float old_y2 = *y2;
    
    int alignedPatchSize=simd_step*((width*height+simd_step)/simd_step);
    
    
    float *buffers=(float*)malloc(9*alignedPatchSize*sizeof(float));
    memset(buffers,0,9*alignedPatchSize*sizeof(float));
    
    float *patch1=buffers;
    float *gx1=&patch1[alignedPatchSize];
    float *gy1=&gx1[alignedPatchSize];
    
    float *patch2=&gy1[alignedPatchSize];
    float *gx2=&patch2[alignedPatchSize];
    float *gy2=&gx2[alignedPatchSize];
    
    float *patchdiff=&gy2[alignedPatchSize];
    float *gx=&patchdiff[alignedPatchSize];
    float *gy=&gx[alignedPatchSize];
    
    memcpy(patch1,img1.data,img1.cols*img1.rows*sizeof(float));
    memcpy(gx1,gradx1.data,gradx1.cols*gradx1.rows*sizeof(float));
    memcpy(gy1,grady1.data,grady1.cols*grady1.rows*sizeof(float));
    
    
    float weights2[4];
    /* Iteratively update the window position */
    do  {
        
        if (*x2-hw < 0.0f || nc2-(*x2+hw) < one_plus_eps ||
            *y2-hh < 0.0f || nr2-(*y2+hh) < one_plus_eps) {
            status = KLT_OOB;
            break;
        }
        
        int xt2,yt2;
        computeBilinearWeight(*x2,*y2,xt2,yt2,weights2);
        
        computeBilinearPatch(patch2,img2,xt2,yt2,width,height,alignedPatchSize,weights2);
        computeBilinearPatch(gx2,gradx2,xt2,yt2,width,height,alignedPatchSize,weights2);
        computeBilinearPatch(gy2,grady2,xt2,yt2,width,height,alignedPatchSize,weights2);
        
        
        computeImageDiff(patch1,patch2,patchdiff,width*height);
        computeImageSum(gx1,gx2,gx,width*height);
        computeImageSum(gy1,gy2,gy,width*height);
        
        compute2X2GradientMatrix(gx,gy,alignedPatchSize,gxx,gyy,gxy);
        
        compute2X1ErrorVector(patchdiff,gx,gy,alignedPatchSize,step_factor,ex,ey);
        
        status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
        
        if (status == KLT_SMALL_DET){
            break;
        }
        
        *x2 += dx;
        *y2 += dy;
        iteration++;
        
        
    }  while ((fabs(dx)>=th || fabs(dy)>=th) && iteration < max_iterations);
    
    
    /* Check whether window is out of bounds */
    if (*x2-hw < 0.0f || nc2-(*x2+hw) < one_plus_eps ||
        *y2-hh < 0.0f || nr2-(*y2+hh) < one_plus_eps)
        status = KLT_OOB;
    
    /* Check whether feature point has moved to much during iteration*/
    if ( (*x2-old_x2) > mdd || (*y2-old_y2) > mdd )
        status = KLT_OOB;
    
    
    /* Check whether residue is too large */
    if (status == KLT_TRACKED)  {
        
        int xt2,yt2;
        computeBilinearWeight(*x2,*y2,xt2,yt2,weights2);
        computeBilinearPatch(patch2,img2,xt2,yt2,width,height,alignedPatchSize,weights2);
        computeImageDiff(patch1,patch2,patchdiff,alignedPatchSize);
        
        //printf("residual %f\n",computeABSImageDiff(patchdiff,alignedPatchSize)/(width*height));
        if (computeABSImageDiff(patchdiff,alignedPatchSize)/(width*height)> max_residue){
            status = KLT_LARGE_RESIDUE;
        }
    }
    
    free(buffers);
    return status;
}





class KLTInvoker{
private:
    
    std::vector<cv::Mat>* pyramid1;
    std::vector<cv::Mat>* pyramid2;
    KLT_TrackingContext   tc;
    KLT_FeatureList       featurelist;
    float                 subsampling;

public:
    
    KLTInvoker(std::vector<cv::Mat>* _pyramid1,
               std::vector<cv::Mat>* _pyramid2,
               KLT_TrackingContext _tc,
               KLT_FeatureList _featurelist,
               float _subsampling){
        
        pyramid1=_pyramid1;
        pyramid2=_pyramid2;
        tc=_tc;
        featurelist=_featurelist;
        subsampling=_subsampling;
    };
    
    void operator ()(const tbb::blocked_range<size_t>& range) const;
};

void KLTInvoker::operator()(const tbb::blocked_range<size_t>& range) const{
    
    for (int indx = range.begin() ; indx < range.end() ; indx++)  {
        
        /* Only track features that are not lost */
        if (featurelist->feature[indx]->val >= 0)  {
            
            
            float xloc = featurelist->feature[indx]->x;
            float yloc = featurelist->feature[indx]->y;
            
            /* Transform location to coarsest resolution */
            for (int r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
                xloc /= subsampling;  yloc /= subsampling;
            }
            
            float xlocout,ylocout;
            
            if(compensate_motion){
                cv::Mat ptMat(3,1,CV_64FC1);
                ptMat.at<double>(0)=subsampling*xloc;
                ptMat.at<double>(1)=subsampling*yloc;
                ptMat.at<double>(2)=1.0;
                ptMat=homography*ptMat;
                xlocout = (ptMat.at<double>(0)/ptMat.at<double>(2))/subsampling;
                ylocout = (ptMat.at<double>(1)/ptMat.at<double>(2))/subsampling;
            }else{
                xlocout =xloc;
                ylocout =yloc;
            }
            
            
            int val;
            /* Beginning with coarsest resolution, do ... */
            for (int r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
                //printf("%d %d\n",indx,r);
                /* Track feature at current resolution */
                xloc *= subsampling;  yloc *= subsampling;
                xlocout *= subsampling;  ylocout *= subsampling;
                
                val = _trackFeature(xloc, yloc,
                                    &xlocout, &ylocout,
                                    (*pyramid1)[3*r],
                                    (*pyramid1)[3*r+1],(*pyramid1)[3*r+2],
                                    (*pyramid2)[3*r],
                                    (*pyramid2)[3*r+1],(*pyramid2)[3*r+2],
                                    tc->window_width, tc->window_height,
                                    tc->step_factor,
                                    tc->max_iterations,
                                    tc->min_determinant,
                                    tc->min_displacement,
                                    tc->max_residue,
                                    tc->lighting_insensitive);
                
                if (val==KLT_SMALL_DET || val==KLT_OOB)
                    break;
            }
            
            int ncols=(*pyramid1)[0].cols,nrows=(*pyramid1)[0].rows;
            
            /* Record feature */
            if (val == KLT_OOB) {
                
                featurelist->feature[indx]->x   = -1.0;
                featurelist->feature[indx]->y   = -1.0;
                featurelist->feature[indx]->val = KLT_OOB;
                
            } else if (_outOfBounds(xlocout, ylocout, ncols, nrows, tc->borderx, tc->bordery))  {
                
                featurelist->feature[indx]->x   = -1.0;
                featurelist->feature[indx]->y   = -1.0;
                featurelist->feature[indx]->val = KLT_OOB;
                
            } else if (val == KLT_SMALL_DET)  {
                featurelist->feature[indx]->x   = -1.0;
                featurelist->feature[indx]->y   = -1.0;
                featurelist->feature[indx]->val = KLT_SMALL_DET;
                
            } else if (val == KLT_LARGE_RESIDUE)  {
                featurelist->feature[indx]->x   = -1.0;
                featurelist->feature[indx]->y   = -1.0;
                featurelist->feature[indx]->val = KLT_LARGE_RESIDUE;
            } else if (val == KLT_MAX_ITERATIONS)  {
                featurelist->feature[indx]->x   = -1.0;
                featurelist->feature[indx]->y   = -1.0;
                featurelist->feature[indx]->val = KLT_MAX_ITERATIONS;
            } else  {
                featurelist->feature[indx]->x = xlocout;
                featurelist->feature[indx]->y = ylocout;
                featurelist->feature[indx]->val = KLT_TRACKED;
                
                /*if(featurelist->feature[indx]->aff_x==-1.0||featurelist->feature[indx]->aff_y==-1.0){
                    
                    if(featurelist->feature[indx]->aff_img==NULL){
                        featurelist->feature[indx]->aff_img=new cv::Mat(tc->affine_window_height,tc->affine_window_width,CV_32F);
                        featurelist->feature[indx]->aff_gradx=new cv::Mat(tc->affine_window_height,tc->affine_window_width,CV_32F);
                        featurelist->feature[indx]->aff_grady=new cv::Mat(tc->affine_window_height,tc->affine_window_width,CV_32F);
                    }
                    
                    cv::Mat((*pyramid1)[0],cv::Rect(int(xloc-tc->affine_window_width/2),
                                                    int(yloc-tc->affine_window_height/2),
                                                    tc->affine_window_width,tc->affine_window_height)).copyTo(*featurelist->feature[indx]->aff_img);
                    
                    cv::Mat((*pyramid1)[1],cv::Rect(int(xloc-tc->affine_window_width/2),
                                                    int(yloc-tc->affine_window_height/2),
                                                    tc->affine_window_width,tc->affine_window_height)).copyTo(*featurelist->feature[indx]->aff_gradx);
                    
                    cv::Mat((*pyramid1)[2],cv::Rect(int(xloc-tc->affine_window_width/2),
                                                    int(yloc-tc->affine_window_height/2),
                                                    tc->affine_window_width,tc->affine_window_height)).copyTo(*featurelist->feature[indx]->aff_grady);
                    
                    featurelist->feature[indx]->aff_x=featurelist->feature[indx]->aff_y=0.0;
                    
                    
                    
                }else{
                    
                    
                    
                    int val=refineFeatureTranslation(&xlocout,&ylocout,
                                                     *featurelist->feature[indx]->aff_img,
                                                     *featurelist->feature[indx]->aff_gradx,
                                                     *featurelist->feature[indx]->aff_grady,
                                                     (*pyramid2)[0],(*pyramid2)[1],(*pyramid2)[2],
                                                     tc->affine_window_width,tc->affine_window_height,
                                                     tc->step_factor,tc->affine_max_iterations,tc->min_determinant,
                                                     tc->min_displacement,tc->affine_max_residue,tc->affine_max_displacement_differ);
                    
                    
                    featurelist->feature[indx]->val = val;
                    
                    if(val != KLT_TRACKED){
                        featurelist->feature[indx]->x   = -1.0;
                        featurelist->feature[indx]->y   = -1.0;
                        featurelist->feature[indx]->aff_x = -1.0;
                        featurelist->feature[indx]->aff_y = -1.0;
                    }else{
                        
                    }
                }*/

            }
            
        }
    }
};

void KLTTrackFeatures(KLT_TrackingContext tc,
                      std::vector<cv::Mat> &pyramid1,
                      std::vector<cv::Mat> &pyramid2,
                      KLT_FeatureList featurelist,const Eigen::Matrix3d &invK)
{
    
    float subsampling = (float) tc->subsampling;
    int ncols=pyramid1[0].cols,nrows=pyramid1[0].rows;
    
    /* Check window size (and correct if necessary) */
    if (tc->window_width % 2 != 1) {
        tc->window_width = tc->window_width+1;
        KLTWarning("Tracking context's window width must be odd.  "
                   "Changing to %d.\n", tc->window_width);
    }
    if (tc->window_height % 2 != 1) {
        tc->window_height = tc->window_height+1;
        KLTWarning("Tracking context's window height must be odd.  "
                   "Changing to %d.\n", tc->window_height);
    }
    
    if (tc->affine_window_width % 2 != 1) {
        tc->affine_window_width = tc->affine_window_width+1;
        KLTWarning("Tracking context's window width must be odd.  "
                   "Changing to %d.\n", tc->affine_window_width);
    }
    if (tc->affine_window_height % 2 != 1) {
        tc->affine_window_height = tc->affine_window_height+1;
        KLTWarning("Tracking context's window height must be odd.  "
                   "Changing to %d.\n", tc->affine_window_height);
    }
    
    if (tc->window_width < 3) {
        tc->window_width = 3;
        KLTWarning("Tracking context's window width must be at least three.  \n"
                   "Changing to %d.\n", tc->window_width);
    }
    if (tc->window_height < 3) {
        tc->window_height = 3;
        KLTWarning("Tracking context's window height must be at least three.  \n"
                   "Changing to %d.\n", tc->window_height);
    }
    
    tbb::parallel_for(tbb::blocked_range<size_t>(0,featurelist->nFeatures),
                      KLTInvoker(&pyramid1,&pyramid2,tc,featurelist,subsampling));
    
    std::vector<cv::Point2f> pts1,pts2;
    for (int i=0;i<featurelist->nFeatures;i++) {
        
        if(featurelist->feature[i]->val!=KLT_TRACKED){
            continue;
        }
        
        Eigen::Vector3d pt((double)featurelist->feature[i]->x,(double)featurelist->feature[i]->y,1.0);
        pt=invK*pt;
        
        featurelist->feature[i]->norm=pt;
        featurelist->feature[i]->norm.normalize();
        
        
        featurelist->feature[i]->vec=featurelist->feature[i]->norm;
        featurelist->feature[i]->vec/=featurelist->feature[i]->vec(2);
        
    }
    featurelist->isOutlierRejected=false;
    
    if (KLT_verbose >= 1)  {
        fprintf(stderr,  "\n\t%d features successfully tracked.\n",
                KLTCountRemainingFeatures(featurelist));
        if (tc->writeInternalImages)
            fprintf(stderr,  "\tWrote images to 'kltimg_tf*.pgm'.\n");
        fflush(stderr);
    }
    
}
#include <fstream>
void compensateLightingAndMotion(const std::vector<cv::Point2f>& pts1,
                                 const std::vector<cv::Point2f>& pts2,
                                 const std::vector<float>& intensity1,
                                 const std::vector<float>& intensity2){
    //compensate motion
    int good_count;
    std::vector<bool> status;
    homography=GSLAM::EstimateHomography(pts1,pts2,status,good_count);
    
    //compensate lighting
    double error0=0;
    double error=0;
    double light_error0=0.0;
    double light_error=0.0;;
    
    int    inlier_count=0;
    Eigen::MatrixXd A=Eigen::MatrixX2d(good_count,2);
    Eigen::VectorXd b=Eigen::VectorXd(good_count);
    
    for(int i=0;i<pts1.size();i++){
        if(status[i]==true){
            inlier_count++;
            A(inlier_count-1,0)=intensity1[i];
            A(inlier_count-1,1)=1.0;
            b(inlier_count-1)=intensity2[i];
            light_error0+=fabs(intensity1[i]-intensity2[i]);
        }
    }
    
    Eigen::MatrixXd AtA=A.transpose()*A;
    Eigen::VectorXd Atb=A.transpose()*b;
    Eigen::VectorXd solution=AtA.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Atb);
    lighting_alpha=(float)solution(0),lighting_beta=(float)solution(1);
    
    for(int i=0;i<pts1.size();i++){
        if(status[i]==true){
            light_error+=fabs(lighting_alpha*intensity1[i]+lighting_beta-intensity2[i]);
        }
    }
    
    if(light_error>light_error0){
        lighting_alpha=1.0;
        lighting_beta=0.0;
    }
    
    static std::ofstream record("/Users/chaos/Desktop/debug/lighting.txt");
    record<<light_error0/inlier_count<<' '<<light_error/inlier_count<<std::endl;
    
    /*assert(inlier_count==good_count);
    std::cout<<"homography:"<<error/inlier_count<<' '
                            <<error0/inlier_count<<' '
                            <<light_error/inlier_count<<' '
                            <<light_error0/inlier_count<<' '
                            <<inlier_count<<' '<<pts1.size()<<std::endl;
    getchar();*/
}
