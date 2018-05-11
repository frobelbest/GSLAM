//
//  selectGoodFeatures.hpp
//  STARCK
//
//  Created by Chaos on 3/30/16.
//  Copyright © 2016 Chaos. All rights reserved.
//


/*********************************************************************
 * selectGoodFeatures.c
 *
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h> /* malloc(), qsort() */
#include <stdio.h>  /* fflush()          */
#include <string.h> /* memset()          */
#include "opencv2/core/core.hpp"
#include "KLT.h"

#pragma once

#define KLT_NOT_FOUND        -1
//#define USE_DILATE

cv::Size gridSize(20,20);
cv::Size sizeByGrid;
int minFeatureInGrid;



typedef enum {SELECTING_ALL, REPLACING_SOME} selectionMode;


static void _fillFeaturemap(int x, int y,
                            uchar *featuremap,
                            int mindist,
                            int ncols,
                            int nrows){
    int ix, iy;
    
    for (iy = y - mindist ; iy <= y + mindist ; iy++){
        for (ix = x - mindist ; ix <= x + mindist ; ix++)
            if (ix >= 0 && ix < ncols && iy >= 0 && iy < nrows)
                featuremap[iy*ncols+ix] = 1;
    }
}

/*********************************************************************
 * _enforceMinimumDistance
 *
 * Removes features that are within close proximity to better features.
 *
 * INPUTS
 * featurelist:  A list of features.  The nFeatures property
 *               is used.
 *
 * OUTPUTS
 * featurelist:  Is overwritten.  Nearby "redundant" features are removed.
 *               Writes -1's into the remaining elements.
 *
 * RETURNS
 * The number of remaining features.
 
 
 */

static int _comparePoints(const void *a, const void *b)
{
    int v1 = *(((int *) a) + 2);
    int v2 = *(((int *) b) + 2);
    
    if (v1 > v2)  return(-1);
    else if (v1 < v2)  return(1);
    else return(0);
}

static void _sortPointList(
                           int *pointlist,
                           int npoints){
    qsort(pointlist, npoints, 3*sizeof(int), _comparePoints);
}



static void _enforceMinimumDistance(uchar *featuremap,
                                    int *pointlist,              /* featurepoints */
                                    int npoints,                 /* number of featurepoints */
                                    KLT_FeatureList featurelist, /* features */
                                    int ncols, int nrows,        /* size of images */
                                    int mindist,                 /* min. dist b/w features */
                                    int min_eigenvalue,          /* min. eigenvalue */
                                    bool overwriteAllFeatures)
{
    int indx;          /* Index into features */
    int x, y, val;     /* Location and trackability of pixel under consideration */
    //int *featuremap; /* Boolean array recording proximity of features */
    int *ptr;
    
    /* Cannot add features with an eigenvalue less than one */
    if (min_eigenvalue < 1)
        min_eigenvalue = 1;
    
    /* Allocate memory for feature map and clear it */
    //featuremap = (int *) malloc(ncols * nrows * sizeof(int));
    //memset(featuremap, 0, ncols*nrows*sizeof(int));
    
    /* Necessary because code below works with (mindist-1) */
    mindist--;
    
    /* If we are keeping all old good features, then add them to the featuremap */
    
    /*if (!overwriteAllFeatures)
        for (indx = 0 ; indx < featurelist->nFeatures ; indx++)
            if (featurelist->feature[indx]->val >= 0)  {
                x   = (int) featurelist->feature[indx]->x;
                y   = (int) featurelist->feature[indx]->y;
                _fillFeaturemap(x, y,INT_MAX,featuremap, mindist, ncols, nrows);
            }
    */
    
    /* For each feature point, in descending order of importance, do ... */
    ptr = pointlist;
    indx = 0;
    while (1)  {
        
        /* If we can't add all the points, then fill in the rest
         of the featurelist with -1's */
        if (ptr >= pointlist + 3*npoints)  {
            while (indx < featurelist->nFeatures)  {
                if (overwriteAllFeatures ||
                    featurelist->feature[indx]->val < 0) {
                    featurelist->feature[indx]->x   = -1;
                    featurelist->feature[indx]->y   = -1;
                    featurelist->feature[indx]->val = KLT_NOT_FOUND;
                    
                    /*
                    featurelist->feature[indx]->aff_img = NULL;
                    featurelist->feature[indx]->aff_img_gradx = NULL;
                    featurelist->feature[indx]->aff_img_grady = NULL;
                    featurelist->feature[indx]->aff_x = -1.0;
                    featurelist->feature[indx]->aff_y = -1.0;
                    featurelist->feature[indx]->aff_Axx = 1.0;
                    featurelist->feature[indx]->aff_Ayx = 0.0;
                    featurelist->feature[indx]->aff_Axy = 0.0;
                    featurelist->feature[indx]->aff_Ayy = 1.0;*/
                    
                }
                indx++;
            }
            break;
        }
        
        x   = *ptr++;
        y   = *ptr++;
        val = *ptr++;
        
        /* Ensure that feature is in-bounds */
        assert(x >= 0);
        assert(x < ncols);
        assert(y >= 0);
        assert(y < nrows);
        
        while (!overwriteAllFeatures &&
               indx < featurelist->nFeatures &&
               featurelist->feature[indx]->val >= 0)
            indx++;
        
        if (indx >= featurelist->nFeatures)  break;
        
        /* If no neighbor has been selected, and if the minimum
         eigenvalue is large enough, then add feature to the current list */
        if (!(featuremap[y*ncols+x]) && val >= min_eigenvalue)  {
            
            featurelist->feature[indx]->x   = (float) x;
            featurelist->feature[indx]->y   = (float) y;
            featurelist->feature[indx]->val = (int) val;
            
            /*
             featurelist->feature[indx]->aff_img = NULL;
             featurelist->feature[indx]->aff_img_gradx = NULL;
             featurelist->feature[indx]->aff_img_grady = NULL;
             featurelist->feature[indx]->aff_x = -1.0;
             featurelist->feature[indx]->aff_y = -1.0;
             featurelist->feature[indx]->aff_Axx = 1.0;
             featurelist->feature[indx]->aff_Ayx = 0.0;
             featurelist->feature[indx]->aff_Axy = 0.0;
             featurelist->feature[indx]->aff_Ayy = 1.0;*/
            indx++;
            
            /* Fill in surrounding region of feature map, but
             make sure that pixels are in-bounds */
            _fillFeaturemap(x, y,featuremap, mindist, ncols, nrows);
        }
    }
    
    /* Free feature map  */
    //free(featuremap);
}



/*********************************************************************
 * _sortPointList
 */



/*********************************************************************
 * _minEigenvalue
 *
 * Given the three distinct elements of the symmetric 2x2 matrix
 *                     [gxx gxy]
 *                     [gxy gyy],
 * Returns the minimum eigenvalue of the matrix.
 */

static float _minEigenvalue(float gxx, float gxy, float gyy)
{
    return (float) ((gxx + gyy - std::sqrt((gxx - gyy)*(gxx - gyy) + 4*gxy*gxy))/2.0f);
}

struct greaterThanPtr :
public std::binary_function<const float *, const float *, bool>
{
    bool operator () (const float * a, const float * b) const
    { return *a > *b; }
};

/*********************************************************************/

void _KLTSelectGoodFeatures(KLT_TrackingContext tc,
                            const cv::Mat& sumDxx,
                            const cv::Mat& sumDyy,
                            const cv::Mat& sumDxy,
                            KLT_FeatureList featurelist,
                            selectionMode mode)
{
    int npoints = 0;
    bool overwriteAllFeatures = (mode == SELECTING_ALL) ? true : false;
    bool floatimages_created = false;
    
    int ncols=sumDxx.cols-1,nrows=sumDxx.rows-1;
    

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
    int window_hw = tc->window_width/2;
    int window_hh = tc->window_height/2;
    
    /* Create pointlist, which is a simplified version of a featurelist, */
    /* for speed.  Contains only integer locations and values. */
    int *pointlist = (int *) malloc(ncols * nrows * 3 * sizeof(int));
    

    uchar *featuremap=(uchar*)malloc(ncols*nrows);
    memset(featuremap,0,ncols*nrows);
    
    if (!overwriteAllFeatures){
        for (int indx = 0 ; indx < featurelist->nFeatures ; indx++){
            if (featurelist->feature[indx]->val >= 0)  {
                int x   = (int) featurelist->feature[indx]->x;
                int y   = (int) featurelist->feature[indx]->y;
                _fillFeaturemap(x, y,featuremap,tc->mindist, ncols, nrows);
            }
        }
    }
    
#ifdef USE_DILATE
    cv::Mat eig=cv::Mat::zeros(nrows,ncols,CV_32F);
#endif
    
    int nSkippedPixels=tc->nSkippedPixels;
    
    /* Compute trackability of each image pixel as the minimum
     of the two eigenvalues of the Z matrix */
    {
        float gxx, gxy, gyy;
        int *ptr;
        
        float val;
        unsigned int limit = 1;
        
        int borderx = tc->borderx;	/* Must not touch cols */
        int bordery = tc->bordery;	/* lost by convolution */
        
        if (borderx < window_hw){
            borderx = window_hw;
        }
        
        if (bordery < window_hh){
            bordery = window_hh;
        }
        
        /* Find largest value of an int */
        for (int i = 0 ; i < sizeof(int) ; i++){
            limit *= 256;
        }
        limit = limit/2 - 1;
        
        /* For most of the pixels in the image, do ... */
        ptr = pointlist;
        for (int y = bordery ; y < nrows - bordery ; y += nSkippedPixels + 1){
            
            float* dxyPtrs[2]={(float*)sumDxy.ptr(y-window_hh),(float*)sumDxy.ptr(y+window_hh+1)};
            float* dxxPtrs[2]={(float*)sumDxx.ptr(y-window_hh),(float*)sumDxx.ptr(y+window_hh+1)};
            float* dyyPtrs[2]={(float*)sumDyy.ptr(y-window_hh),(float*)sumDyy.ptr(y+window_hh+1)};
            
#ifdef USE_DILATE
            float* eigPtr=(float*)eig.ptr(y);
#endif
            
            for (int x = borderx,x1=borderx-window_hw,x2=borderx+window_hw+1;
                     x < ncols - borderx ;
                     x += nSkippedPixels + 1,x1+=nSkippedPixels + 1,x2+=nSkippedPixels + 1)  {
                
                if (featuremap[y*ncols+x]) {
                    continue;
                }
                /* Sum the gradients in the surrounding window */
                gxy=dxyPtrs[0][x1]+dxyPtrs[1][x2]-dxyPtrs[1][x1]-dxyPtrs[0][x2];
                gxx=dxxPtrs[0][x1]+dxxPtrs[1][x2]-dxxPtrs[1][x1]-dxxPtrs[0][x2];
                gyy=dyyPtrs[0][x1]+dyyPtrs[1][x2]-dyyPtrs[1][x1]-dyyPtrs[0][x2];
                
                /* Store the trackability of the pixel as the minimum
                 of the two eigenvalues */
                float val= _minEigenvalue(gxx, gxy, gyy);
                
                
                if (val > limit)  {
                    val = (float) limit;
                }
                
                if (val>tc->min_eigenvalue){
#ifdef USE_DILATE
                    eigPtr[x] = val;
#else
                    *ptr++ = x;
                    *ptr++ = y;
                    *ptr++ = (int) val;
                    npoints++;
#endif
                }
            }
        }
    }

#ifdef USE_DILATE
    
    cv::Mat tmp;
    cv::dilate(eig,tmp,cv::Mat());
    cv::Size imgsize=tmp.size();
    
    std::vector<const float*> tmpCorners;
    for( int y = 1; y < imgsize.height - 1; y++ )
    {
        const float* eig_data = (const float*)eig.ptr(y);
        const float* tmp_data = (const float*)tmp.ptr(y);

        for( int x = 1; x < imgsize.width - 1; x++ )
        {
            float val = eig_data[x];
            if( val != 0&& val == tmp_data[x])
                tmpCorners.push_back(eig_data + x);
        }
    }
    
    std::sort( tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());
    
    npoints=tmpCorners.size();
    for (int i=0;i<npoints;i++) {
        int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
        pointlist[3*i+2] = (int)(*tmpCorners[i]);
        pointlist[3*i+1] = (int)(ofs / eig.step);
        pointlist[3*i]   = (int)((ofs-pointlist[3*i+1]*eig.step)/sizeof(float));
    }
    
#else
    
    _sortPointList(pointlist, npoints);
    
#endif

    
    /* Enforce minimum distance between features */
    _enforceMinimumDistance(featuremap,
                            pointlist,
                            npoints,
                            featurelist,
                            ncols, nrows,
                            tc->mindist,
                            tc->min_eigenvalue,
                            overwriteAllFeatures);
    
    /* Free memory */
    free(featuremap);
    free(pointlist);
}




void KLTSelectGoodFeatures(KLT_TrackingContext tc,
                           const cv::Mat&  sumDxx,
                           const cv::Mat&  sumDyy,
                           const cv::Mat&  sumDxy,
                           KLT_FeatureList fl){
    int ncols=sumDxx.cols-1;
    int nrows=sumDxx.rows-1;
    
    _KLTSelectGoodFeatures(tc,sumDxx,sumDyy,sumDxy,fl, SELECTING_ALL);
}


void KLTReplaceLostFeatures(KLT_TrackingContext tc,
                            const cv::Mat& sumDxx,
                            const cv::Mat& sumDyy,
                            const cv::Mat& sumDxy,
                            KLT_FeatureList fl){
    
    int nLostFeatures = fl->nFeatures - KLTCountRemainingFeatures(fl);
    int ncols=sumDxx.cols-1;
    int nrows=sumDxx.rows-1;
    
    if (nLostFeatures > 0)
        _KLTSelectGoodFeatures(tc,sumDxx,sumDyy,sumDxy,fl,REPLACING_SOME);
    
}

typedef enum {SELECTING_ALL2, REPLACING_SOME2} selectionMode2;


static void _fillFeaturemap2(int x, int y,
                             uchar *featuremap,
                             int mindist,
                             int ncols,
                             int nrows){
    int ix, iy;
    
    for (iy = y - mindist ; iy <= y + mindist ; iy++){
        for (ix = x - mindist ; ix <= x + mindist ; ix++)
            if (ix >= 0 && ix < ncols && iy >= 0 && iy < nrows)
                featuremap[iy*ncols+ix] = 1;
    }
}

/*********************************************************************
 * _enforceMinimumDistance
 *
 * Removes features that are within close proximity to better features.
 *
 * INPUTS
 * featurelist:  A list of features.  The nFeatures property
 *               is used.
 *
 * OUTPUTS
 * featurelist:  Is overwritten.  Nearby "redundant" features are removed.
 *               Writes -1's into the remaining elements.
 *
 * RETURNS
 * The number of remaining features.
 
 
 */

static int _comparePoints2(const void *a, const void *b)
{
    int v1 = *(((int *) a) + 2);
    int v2 = *(((int *) b) + 2);
    
    if (v1 > v2)  return(-1);
    else if (v1 < v2)  return(1);
    else return(0);
}

static void _sortPointList2(
                            int *pointlist,
                            int npoints){
    qsort(pointlist, npoints, 3*sizeof(int), _comparePoints2);
}

static float _minEigenvalue2(float gxx, float gxy, float gyy)
{
    return (float) ((gxx + gyy - std::sqrt((gxx - gyy)*(gxx - gyy) + 4*gxy*gxy))/2.0f);
}
void _enforceMinimumDistance2(uchar* featuremap,
                              int *pointlist,              /* featurepoints */
                              int npoints,                 /* number of featurepoints */
                              KLT_FeatureList featurelist, /* features */
                              int ncols, int nrows,        /* size of images */
                              int mindist,                 /* min. dist b/w features */
                              int min_eigenvalue,          /* min. eigenvalue */
                              bool overwriteAllFeatures)
{
    int indx;          /* Index into features */
    int x, y, val;     /* Location and trackability of pixel under consideration */
    /* Boolean array recording proximity of features */
    int *ptr;
    
    /* Cannot add features with an eigenvalue less than one */
    if (min_eigenvalue < 1)
        min_eigenvalue = 1;
    
     //mindist--;
    /* Allocate memory for feature map and clear it */
    
    /*
     featuremap = (uchar *) malloc(ncols * nrows * sizeof(uchar));
     memset(featuremap, 0, ncols*nrows);
     
    
     
     if (!overwriteAllFeatures)
     for (indx = 0 ; indx < featurelist->nFeatures ; indx++)
     if (featurelist->feature[indx]->val >= 0)  {
     x   = (int) featurelist->feature[indx]->x;
     y   = (int) featurelist->feature[indx]->y;
     _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);
     }
     */
    
    /* For each feature point, in descending order of importance, do ... */
    ptr = pointlist;
    indx = 0;
    while (1)  {
        
        /* If we can't add all the points, then fill in the rest
         of the featurelist with -1's */
        if (ptr >= pointlist + 3*npoints)  {
            while (indx < featurelist->nFeatures)  {
                if (overwriteAllFeatures ||
                    featurelist->feature[indx]->val < 0) {
                    featurelist->feature[indx]->x   = -1;
                    featurelist->feature[indx]->y   = -1;
                    featurelist->feature[indx]->val = KLT_NOT_FOUND;
                    
                    /*featurelist->feature[indx]->aff_img = NULL;
                    featurelist->feature[indx]->aff_img_gradx = NULL;
                    featurelist->feature[indx]->aff_img_grady = NULL;*/
                    
                    featurelist->feature[indx]->aff_x = -1.0;
                    featurelist->feature[indx]->aff_y = -1.0;
                    featurelist->feature[indx]->aff_Axx = 1.0;
                    featurelist->feature[indx]->aff_Ayx = 0.0;
                    featurelist->feature[indx]->aff_Axy = 0.0;
                    featurelist->feature[indx]->aff_Ayy = 1.0;
                }
                indx++;
            }
            break;
        }
        
        x   = *ptr++;
        y   = *ptr++;
        val = *ptr++;
        
        /* Ensure that feature is in-bounds */
        assert(x >= 0);
        assert(x < ncols);
        assert(y >= 0);
        assert(y < nrows);
        
        while (!overwriteAllFeatures &&
               indx < featurelist->nFeatures &&
               featurelist->feature[indx]->val >= 0)
            indx++;
        
        if (indx >= featurelist->nFeatures)  break;
        
        /* If no neighbor has been selected, and if the minimum
         eigenvalue is large enough, then add feature to the current list */
        if (!featuremap[y*ncols+x] && val >= min_eigenvalue)  {
            featurelist->feature[indx]->x   = (KLT_locType) x;
            featurelist->feature[indx]->y   = (KLT_locType) y;
            featurelist->feature[indx]->val = (int) val;
            
            /*featurelist->feature[indx]->aff_img = NULL;
            featurelist->feature[indx]->aff_gradx = NULL;
            featurelist->feature[indx]->aff_grady = NULL;*/
            
            featurelist->feature[indx]->aff_x = -1.0;
            featurelist->feature[indx]->aff_y = -1.0;
            featurelist->feature[indx]->aff_Axx = 1.0;
            featurelist->feature[indx]->aff_Ayx = 0.0;
            featurelist->feature[indx]->aff_Axy = 0.0;
            featurelist->feature[indx]->aff_Ayy = 1.0;
            indx++;
            
            /* Fill in surrounding region of feature map, but
             make sure that pixels are in-bounds */
            _fillFeaturemap2(x, y, featuremap, mindist, ncols, nrows);
        }
    }
    
    /* Free feature map  */
    //free(featuremap);
}

void _KLTSelectGoodFeatures(KLT_TrackingContext tc,
                            const cv::Mat& sumDxx,
                            const cv::Mat& sumDyy,
                            const cv::Mat& sumDxy,
                            KLT_FeatureList featurelist,
                            selectionMode2 mode)
{
    int npoints = 0;
    bool overwriteAllFeatures = (mode == SELECTING_ALL2) ? true : false;
    bool floatimages_created = false;
    
    int ncols=sumDxx.cols-1,nrows=sumDxx.rows-1;
    
    
    
    /* Check window size (and correct if necessary) */
    if (tc->window_width % 2 != 1) {
        tc->window_width = tc->window_width+1;
    }
    if (tc->window_height % 2 != 1) {
        tc->window_height = tc->window_height+1;
        
    }
    if (tc->window_width < 3) {
        tc->window_width = 3;
        
    }
    if (tc->window_height < 3) {
        tc->window_height = 3;
        
    }
    int window_hw = tc->window_width/2;
    int window_hh = tc->window_height/2;
    
    /* Create pointlist, which is a simplified version of a featurelist, */
    /* for speed.  Contains only integer locations and values. */
    int *pointlist = (int *) malloc(ncols * nrows * 3 * sizeof(int));
    
    
    
    
    uchar *featuremap=(uchar*)malloc(ncols*nrows);
    memset(featuremap,0,ncols*nrows);
    
    if (!overwriteAllFeatures){
        for (int indx = 0 ; indx < featurelist->nFeatures ; indx++){
            if (featurelist->feature[indx]->val >= 0)  {
                int x   = (int) featurelist->feature[indx]->x;
                int y   = (int) featurelist->feature[indx]->y;
                _fillFeaturemap2(x, y,featuremap,tc->mindist, ncols, nrows);
            }
        }
    }
#ifdef USE_DILATE
    cv::Mat eig=cv::Mat::zeros(nrows,ncols,CV_32F);
#endif
    
    /* Compute trackability of each image pixel as the minimum
     of the two eigenvalues of the Z matrix */
    {
        float gxx, gxy, gyy;
        int *ptr;
        
        float val;
        unsigned int limit = 1;
        
        int borderx = tc->borderx;	/* Must not touch cols */
        int bordery = tc->bordery;	/* lost by convolution */
        
        if (borderx < window_hw){
            borderx = window_hw;
        }
        
        if (bordery < window_hh){
            bordery = window_hh;
        }
        
        /* Find largest value of an int */
        for (int i = 0 ; i < sizeof(int) ; i++){
            limit *= 256;
        }
        limit = limit/2 - 1;
        
        /* For most of the pixels in the image, do ... */
        ptr = pointlist;
        int nSkippedPixels=tc->nSkippedPixels;
        
        for (int y = bordery ; y < nrows - bordery ; y += nSkippedPixels + 1){
            
            float* dxyPtrs[2]={(float*)sumDxy.ptr(y-window_hh),(float*)sumDxy.ptr(y+window_hh+1)};
            float* dxxPtrs[2]={(float*)sumDxx.ptr(y-window_hh),(float*)sumDxx.ptr(y+window_hh+1)};
            float* dyyPtrs[2]={(float*)sumDyy.ptr(y-window_hh),(float*)sumDyy.ptr(y+window_hh+1)};
            
#ifdef USE_DILATE
            float* eigPtr=(float*)eig.ptr(y);
#endif
            
            for (int x = borderx,x1=borderx-window_hw,x2=borderx+window_hw+1;
                 x < ncols - borderx ;
                 x += nSkippedPixels + 1,x1+=nSkippedPixels + 1,x2+=nSkippedPixels + 1)  {
                
                if (featuremap[y*ncols+x]==1) {
                    continue;
                }
                /* Sum the gradients in the surrounding window */
                gxy=dxyPtrs[0][x1]+dxyPtrs[1][x2]-dxyPtrs[1][x1]-dxyPtrs[0][x2];
                gxx=dxxPtrs[0][x1]+dxxPtrs[1][x2]-dxxPtrs[1][x1]-dxxPtrs[0][x2];
                gyy=dyyPtrs[0][x1]+dyyPtrs[1][x2]-dyyPtrs[1][x1]-dyyPtrs[0][x2];
                
                /* Store the trackability of the pixel as the minimum
                 of the two eigenvalues */
                float _val= _minEigenvalue2(gxx, gxy, gyy);
                int val;
                
                if (_val > limit)  {
                    val = limit;
                }else{
                    val=(int)(_val+0.5);
                }
                
                if (val>tc->min_eigenvalue){
#ifdef USE_DILATE
                    eigPtr[x] = val;
#else
                    *ptr++ = x;
                    *ptr++ = y;
                    *ptr++ = val;
                    npoints++;
#endif
                }
            }
        }
    }
    //printf("%d noints\n",npoints);
#ifdef USE_DILATE
    
    cv::Mat tmp;
    cv::dilate(eig,tmp,cv::Mat());
    cv::Size imgsize=tmp.size();
    
    std::vector<const float*> tmpCorners;
    for( int y = 1; y < imgsize.height - 1; y++ )
    {
        const float* eig_data = (const float*)eig.ptr(y);
        const float* tmp_data = (const float*)tmp.ptr(y);
        
        for( int x = 1; x < imgsize.width - 1; x++ )
        {
            float val = eig_data[x];
            if( val != 0&& val == tmp_data[x])
                tmpCorners.push_back(eig_data + x);
        }
    }
    
    std::sort( tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());
    
    npoints=tmpCorners.size();
    for (int i=0;i<npoints;i++) {
        int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
        pointlist[3*i+2] = (int)(*tmpCorners[i]);
        pointlist[3*i+1] = (int)(ofs / eig.step);
        pointlist[3*i]   = (int)((ofs-pointlist[3*i+1]*eig.step)/sizeof(float));
    }
    
#else
    
    _sortPointList2(pointlist, npoints);
    
    
    
#endif
    
    /* Enforce minimum distance between features */
    _enforceMinimumDistance2(featuremap,
                             pointlist,
                             npoints,
                             featurelist,
                             ncols, nrows,
                             tc->mindist,
                             tc->min_eigenvalue,
                             overwriteAllFeatures);
    
    
    /* Free memory */
    free(featuremap);
    free(pointlist);
}

void _KLTSelectGoodFeatures2(KLT_TrackingContext tc,
                            const cv::Mat& sumDxx,
                            const cv::Mat& sumDyy,
                            const cv::Mat& sumDxy,
                            KLT_FeatureList featurelist,
                            selectionMode mode)
{
    int npoints = 0;
    bool overwriteAllFeatures = (mode == SELECTING_ALL) ? true : false;
    bool floatimages_created = false;
    
    int ncols=sumDxx.cols-1,nrows=sumDxx.rows-1;
    
    
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
    int window_hw = tc->window_width/2;
    int window_hh = tc->window_height/2;
    
    /* Create pointlist, which is a simplified version of a featurelist, */
    /* for speed.  Contains only integer locations and values. */
    int *pointlist = (int *) malloc(ncols * nrows * 3 * sizeof(int));
    
    
    uchar *featuremap=(uchar*)malloc(ncols*nrows);
    memset(featuremap,0,ncols*nrows);
    
    if (!overwriteAllFeatures){
        for (int indx = 0 ; indx < featurelist->nFeatures ; indx++){
            if (featurelist->feature[indx]->val >= 0)  {
                int x   = (int) featurelist->feature[indx]->x;
                int y   = (int) featurelist->feature[indx]->y;
                _fillFeaturemap(x, y,featuremap,tc->mindist, ncols, nrows);
            }
        }
    }
    
#ifdef USE_DILATE
    cv::Mat eig=cv::Mat::zeros(nrows,ncols,CV_32F);
#endif
    
    int nSkippedPixels=tc->nSkippedPixels;
    
    /* Compute trackability of each image pixel as the minimum
     of the two eigenvalues of the Z matrix */
    {
        float gxx, gxy, gyy;
        int *ptr;
        
        float val;
        unsigned int limit = 1;
        
        int borderx = tc->borderx;	/* Must not touch cols */
        int bordery = tc->bordery;	/* lost by convolution */
        
        if (borderx < window_hw){
            borderx = window_hw;
        }
        
        if (bordery < window_hh){
            bordery = window_hh;
        }
        
        /* Find largest value of an int */
        for (int i = 0 ; i < sizeof(int) ; i++){
            limit *= 256;
        }
        limit = limit/2 - 1;
        
        /* For most of the pixels in the image, do ... */
        ptr = pointlist;
        for (int y = bordery ; y < nrows - bordery ; y += nSkippedPixels + 1){
            
            float* dxyPtrs[2]={(float*)sumDxy.ptr(y-window_hh),(float*)sumDxy.ptr(y+window_hh+1)};
            float* dxxPtrs[2]={(float*)sumDxx.ptr(y-window_hh),(float*)sumDxx.ptr(y+window_hh+1)};
            float* dyyPtrs[2]={(float*)sumDyy.ptr(y-window_hh),(float*)sumDyy.ptr(y+window_hh+1)};
            
#ifdef USE_DILATE
            float* eigPtr=(float*)eig.ptr(y);
#endif
            
            for (int x = borderx,x1=borderx-window_hw,x2=borderx+window_hw+1;
                 x < ncols - borderx ;
                 x += nSkippedPixels + 1,x1+=nSkippedPixels + 1,x2+=nSkippedPixels + 1)  {
                
                if (featuremap[y*ncols+x]) {
                    continue;
                }
                /* Sum the gradients in the surrounding window */
                gxy=dxyPtrs[0][x1]+dxyPtrs[1][x2]-dxyPtrs[1][x1]-dxyPtrs[0][x2];
                gxx=dxxPtrs[0][x1]+dxxPtrs[1][x2]-dxxPtrs[1][x1]-dxxPtrs[0][x2];
                gyy=dyyPtrs[0][x1]+dyyPtrs[1][x2]-dyyPtrs[1][x1]-dyyPtrs[0][x2];
                
                /* Store the trackability of the pixel as the minimum
                 of the two eigenvalues */
                float val= _minEigenvalue(gxx, gxy, gyy);
                
                
                if (val > limit)  {
                    val = (float) limit;
                }
                
                if (val>tc->min_eigenvalue){
#ifdef USE_DILATE
                    eigPtr[x] = val;
#else
                    *ptr++ = x;
                    *ptr++ = y;
                    *ptr++ = (int) val;
                    npoints++;
#endif
                }
            }
        }
    }
    
#ifdef USE_DILATE
    
    cv::Mat tmp;
    cv::dilate(eig,tmp,cv::Mat());
    cv::Size imgsize=tmp.size();
    
    std::vector<const float*> tmpCorners;
    for( int y = 1; y < imgsize.height - 1; y++ )
    {
        const float* eig_data = (const float*)eig.ptr(y);
        const float* tmp_data = (const float*)tmp.ptr(y);
        
        for( int x = 1; x < imgsize.width - 1; x++ )
        {
            float val = eig_data[x];
            if( val != 0&& val == tmp_data[x])
                tmpCorners.push_back(eig_data + x);
        }
    }
    
    std::sort( tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());
    
    npoints=tmpCorners.size();
    for (int i=0;i<npoints;i++) {
        int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
        pointlist[3*i+2] = (int)(*tmpCorners[i]);
        pointlist[3*i+1] = (int)(ofs / eig.step);
        pointlist[3*i]   = (int)((ofs-pointlist[3*i+1]*eig.step)/sizeof(float));
    }
    
#else
    
    _sortPointList(pointlist, npoints);
    
#endif
    
    
    /* Enforce minimum distance between features */
    _enforceMinimumDistance(featuremap,
                            pointlist,
                            npoints,
                            featurelist,
                            ncols, nrows,
                            tc->mindist,
                            tc->min_eigenvalue,
                            overwriteAllFeatures);
    
    /* Free memory */
    free(featuremap);
    free(pointlist);
}



void KLTSelectGoodFeatures2(KLT_TrackingContext tc,
                           const cv::Mat&  sumDxx,
                           const cv::Mat&  sumDyy,
                           const cv::Mat&  sumDxy,
                           KLT_FeatureList fl){
    int ncols=sumDxx.cols-1;
    int nrows=sumDxx.rows-1;
    
    _KLTSelectGoodFeatures2(tc,sumDxx,sumDyy,sumDxy,fl,SELECTING_ALL);
}


void KLTReplaceLostFeatures2(KLT_TrackingContext tc,
                            const cv::Mat& sumDxx,
                            const cv::Mat& sumDyy,
                            const cv::Mat& sumDxy,
                            KLT_FeatureList fl){
    
    int nLostFeatures = fl->nFeatures - KLTCountRemainingFeatures(fl);
    int ncols=sumDxx.cols-1;
    int nrows=sumDxx.rows-1;
    
    if (nLostFeatures > 0)
        _KLTSelectGoodFeatures(tc,sumDxx,sumDyy,sumDxy,fl,REPLACING_SOME2);
    
}



struct greaterThanPtrInt:
public std::binary_function<const int *, const int *, bool>
{
    bool operator () (const int * a, const int * b) const
    { return *a > *b; }
};

struct smallerDistance:
public std::binary_function<const int,const int, bool>
{
    bool operator () (const int a, const int b) const{
        
        float distanceY=a/sizeByGrid.width-float(sizeByGrid.height)/2.0;
        float distanceX=a%sizeByGrid.width-float(sizeByGrid.width)/2.0;
        
        float distanceA=distanceX*distanceX+distanceY*distanceY;
        
        distanceY=b/sizeByGrid.width-float(sizeByGrid.height)/2.0;
        distanceX=b%sizeByGrid.width-float(sizeByGrid.width)/2.0;
        
        float distanceB=distanceX*distanceX+distanceY*distanceY;
        
        return distanceA<distanceB;
    }
};


static std::vector<int> gridSequence;

void initializeGridSequence(){
    gridSequence.resize(sizeByGrid.area());
    for(int i=0;i<gridSequence.size();i++){
        gridSequence[i]=i;
    }
    std::sort(gridSequence.begin(),gridSequence.end(),smallerDistance());
}


void _KLTSelectGoodFeaturesUniform(KLT_TrackingContext tc,
                                   const cv::Mat& sumDxx,
                                   const cv::Mat& sumDyy,
                                   const cv::Mat& sumDxy,
                                   KLT_FeatureList featurelist,
                                   selectionMode mode)
{
    int npoints = 0;
    bool overwriteAllFeatures = (mode == SELECTING_ALL) ? true : false;
    bool floatimages_created = false;
    
    int ncols=sumDxx.cols-1,nrows=sumDxx.rows-1;
    
    
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
    int window_hw = tc->window_width/2;
    int window_hh = tc->window_height/2;
    
    /* Create pointlist, which is a simplified version of a featurelist, */
    /* for speed.  Contains only integer locations and values. */
    
    
    
    uchar *featuremap=(uchar*)malloc(ncols*nrows);
    memset(featuremap,0,ncols*nrows);
    
    
    //grid and eigmap
    std::vector<std::vector<int*> > grids;
    grids.resize(sizeByGrid.width*sizeByGrid.height);
    for(int i=0;i<grids.size();i++){
        grids[i].clear();
    }
    
    cv::Mat eig=cv::Mat(nrows,ncols,CV_32S);
    
    
    if (!overwriteAllFeatures){
        for (int indx = 0 ; indx < featurelist->nFeatures ; indx++){
            if (featurelist->feature[indx]->val >= 0)  {
                
                int x   = (int) featurelist->feature[indx]->x;
                int y   = (int) featurelist->feature[indx]->y;
                _fillFeaturemap(x,y,featuremap,tc->mindist, ncols, nrows);
                
                
                //register to grid
                int* eigPtr=(int*)eig.ptr(y);
                eigPtr[x]=(int)INT_MAX;
                
                int yByGrid=y/gridSize.height;
                int xByGrid=x/gridSize.width;
                grids[yByGrid*sizeByGrid.width+xByGrid].push_back(&eigPtr[x]);
            }
        }
    }
    
    int nSkippedPixels=tc->nSkippedPixels;
    

    
    
    /* Compute trackability of each image pixel as the minimum
     of the two eigenvalues of the Z matrix */
    {
        float gxx, gxy, gyy;
        int *ptr;
        
        float val;
        unsigned int limit = 1;
        
        int borderx = tc->borderx;	/* Must not touch cols */
        int bordery = tc->bordery;	/* lost by convolution */
        
        if (borderx < window_hw){
            borderx = window_hw;
        }
        
        if (bordery < window_hh){
            bordery = window_hh;
        }
        
        /* Find largest value of an int */
        for (int i = 0 ; i < sizeof(int) ; i++){
            limit *= 256;
        }
        limit = limit/2 - 1;
        
        for (int y = bordery ; y < nrows - bordery ; y += nSkippedPixels + 1){
            
            float* dxyPtrs[2]={(float*)sumDxy.ptr(y-window_hh),(float*)sumDxy.ptr(y+window_hh+1)};
            float* dxxPtrs[2]={(float*)sumDxx.ptr(y-window_hh),(float*)sumDxx.ptr(y+window_hh+1)};
            float* dyyPtrs[2]={(float*)sumDyy.ptr(y-window_hh),(float*)sumDyy.ptr(y+window_hh+1)};
            
            int* eigPtr=(int*)eig.ptr(y);

            
            for (int x = borderx,x1=borderx-window_hw,x2=borderx+window_hw+1;
                 x < ncols - borderx ;
                 x += nSkippedPixels + 1,x1+=nSkippedPixels + 1,x2+=nSkippedPixels + 1)  {
                
                if (featuremap[y*ncols+x]) {
                    //assert(0);
                    continue;
                }
                /* Sum the gradients in the surrounding window */
                gxy=dxyPtrs[0][x1]+dxyPtrs[1][x2]-dxyPtrs[1][x1]-dxyPtrs[0][x2];
                gxx=dxxPtrs[0][x1]+dxxPtrs[1][x2]-dxxPtrs[1][x1]-dxxPtrs[0][x2];
                gyy=dyyPtrs[0][x1]+dyyPtrs[1][x2]-dyyPtrs[1][x1]-dyyPtrs[0][x2];
                
                /* Store the trackability of the pixel as the minimum
                 of the two eigenvalues */
                float val= _minEigenvalue(gxx, gxy, gyy);
                
                
                if (val > limit)  {
                    assert(0);
                    val = (float) limit;
                }
                
                if (val>tc->min_eigenvalue){
                    eigPtr[x]=(int)val;
                    int yByGrid=y/gridSize.height;
                    int xByGrid=x/gridSize.width;
                    grids[yByGrid*sizeByGrid.width+xByGrid].push_back(&eigPtr[x]);
                }
            }
        }
    }
    
    
    for(int i=0;i<grids.size();i++){
        std::sort(grids[i].begin(),grids[i].end(), greaterThanPtrInt());
    }
    
    
    int indx=0;
    
    for(int ind=0;ind<gridSequence.size();ind++){
        
        int i=gridSequence[ind];
        int featureAdded=0;
        
        for(int j=0;j<grids[i].size();j++){
            
            
            int val=(*grids[i][j]);
            
            if(val==INT_MAX){
                featureAdded++;
            }
            
            /*if(featureAdded>maxFeatureInGrid&&val!=INT_MAX){
                break;
            }*/

            
            if(val!=INT_MAX){
                
                int ofs = (int)((const uchar*)grids[i][j] - eig.ptr());
                int y   = (int)(ofs / eig.step);
                int x   = (int)((ofs-y*eig.step)/sizeof(int));
                
                
                while (!overwriteAllFeatures &&indx < featurelist->nFeatures&& featurelist->feature[indx]->val >= 0){
                    indx++;
                }
                
                if (indx >= featurelist->nFeatures){
                    if(featureAdded>minFeatureInGrid){
                        break;
                    }else{
                        
                        featurelist->nFeatures++;
                        
                        featurelist->feature[indx]->x   = -1;
                        featurelist->feature[indx]->y   = -1;
                        featurelist->feature[indx]->val = -1;
                        
                        featurelist->feature[indx]->aff_x = -1.0;
                        featurelist->feature[indx]->aff_y = -1.0;
                        featurelist->feature[indx]->aff_Axx = 1.0;
                        featurelist->feature[indx]->aff_Ayx = 0.0;
                        featurelist->feature[indx]->aff_Axy = 0.0;
                        featurelist->feature[indx]->aff_Ayy = 1.0;
                        
                    }
                }
                
                
                if (!featuremap[y*ncols+x])  {
                    
                    featurelist->feature[indx]->x   = (KLT_locType) x;
                    featurelist->feature[indx]->y   = (KLT_locType) y;
                    featurelist->feature[indx]->val = (int) val;
                    
                    featurelist->feature[indx]->aff_x = -1.0;
                    featurelist->feature[indx]->aff_y = -1.0;
                    featurelist->feature[indx]->aff_Axx = 1.0;
                    featurelist->feature[indx]->aff_Ayx = 0.0;
                    featurelist->feature[indx]->aff_Axy = 0.0;
                    featurelist->feature[indx]->aff_Ayy = 1.0;
                    indx++;
                    
                    _fillFeaturemap(x, y, featuremap,tc->mindist, ncols, nrows);
                    featureAdded++;
                }
            }
        }
    }
    
    
    while (indx < featurelist->nFeatures)  {
        
        if (overwriteAllFeatures || featurelist->feature[indx]->val < 0) {
            
            featurelist->feature[indx]->x   = -1;
            featurelist->feature[indx]->y   = -1;
            featurelist->feature[indx]->val = KLT_NOT_FOUND;
            
            featurelist->feature[indx]->aff_x = -1.0;
            featurelist->feature[indx]->aff_y = -1.0;
            featurelist->feature[indx]->aff_Axx = 1.0;
            featurelist->feature[indx]->aff_Ayx = 0.0;
            featurelist->feature[indx]->aff_Axy = 0.0;
            featurelist->feature[indx]->aff_Ayy = 1.0;
        }
        indx++;
    }
    
    free(featuremap);
    
}

void KLTSelectGoodFeaturesUniform(
                                  KLT_TrackingContext tc,
                            const cv::Mat&  sumDxx,
                            const cv::Mat&  sumDyy,
                            const cv::Mat&  sumDxy,
                            KLT_FeatureList fl){
    
    
    int ncols=sumDxx.cols-1;
    int nrows=sumDxx.rows-1;
    
    
    sizeByGrid.width=ncols/gridSize.width;
    sizeByGrid.height=nrows/gridSize.height;
    
    minFeatureInGrid=int((fl->nFeatures/float(sizeByGrid.area()))+0.5);
    initializeGridSequence();
    
    
    _KLTSelectGoodFeaturesUniform(tc,sumDxx,sumDyy,sumDxy,fl, SELECTING_ALL);
}


void KLTReplaceLostFeaturesUniform(
                                   KLT_TrackingContext tc,
                             const cv::Mat& sumDxx,
                             const cv::Mat& sumDyy,
                             const cv::Mat& sumDxy,
                             KLT_FeatureList fl){
    
    int nLostFeatures = fl->nFeatures - KLTCountRemainingFeatures(fl);
    int ncols=sumDxx.cols-1;
    int nrows=sumDxx.rows-1;
    
    if (nLostFeatures > 0)
        _KLTSelectGoodFeatures(tc,sumDxx,sumDyy,sumDxy,fl,REPLACING_SOME);
    
}









