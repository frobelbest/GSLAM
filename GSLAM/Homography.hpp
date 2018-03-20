//
//  Homography.hpp
//  GSLAM
//
//  Created by Chaos on 2017-02-09.
//  Copyright © 2017 ctang. All rights reserved.
//

#ifndef Homography_h
#define Homography_h
#pragma once
#include "opencv2/core/core.hpp"

namespace GSLAM{
    
    static void icvGetRTMatrix(const CvPoint2D32f* a,
                        const CvPoint2D32f* b,
                        int count, CvMat* M,
                        int full_affine){
        
        if (full_affine)
        {
            double sa[36], sb[6];
            CvMat A = cvMat(6, 6, CV_64F, sa), B = cvMat(6, 1, CV_64F, sb);
            CvMat MM = cvMat(6, 1, CV_64F, M->data.db);
            
            int i;
            
            memset(sa, 0, sizeof(sa));
            memset(sb, 0, sizeof(sb));
            
            for (i = 0; i < count; i++)
            {
                sa[0] += a[i].x*a[i].x;
                sa[1] += a[i].y*a[i].x;
                sa[2] += a[i].x;
                
                sa[6] += a[i].x*a[i].y;
                sa[7] += a[i].y*a[i].y;
                sa[8] += a[i].y;
                
                sa[12] += a[i].x;
                sa[13] += a[i].y;
                sa[14] += 1;
                
                sb[0] += a[i].x*b[i].x;
                sb[1] += a[i].y*b[i].x;
                sb[2] += b[i].x;
                sb[3] += a[i].x*b[i].y;
                sb[4] += a[i].y*b[i].y;
                sb[5] += b[i].y;
            }
            
            sa[21] = sa[0];
            sa[22] = sa[1];
            sa[23] = sa[2];
            sa[27] = sa[6];
            sa[28] = sa[7];
            sa[29] = sa[8];
            sa[33] = sa[12];
            sa[34] = sa[13];
            sa[35] = sa[14];
            
            cvSolve(&A, &B, &MM, CV_SVD);
        }
        else
        {
            double sa[16], sb[4], m[4], *om = M->data.db;
            CvMat A = cvMat(4, 4, CV_64F, sa), B = cvMat(4, 1, CV_64F, sb);
            CvMat MM = cvMat(4, 1, CV_64F, m);
            
            int i;
            
            memset(sa, 0, sizeof(sa));
            memset(sb, 0, sizeof(sb));
            
            for (i = 0; i < count; i++)
            {
                sa[0] += a[i].x*a[i].x + a[i].y*a[i].y;
                sa[1] += 0;
                sa[2] += a[i].x;
                sa[3] += a[i].y;
                
                sa[4] += 0;
                sa[5] += a[i].x*a[i].x + a[i].y*a[i].y;
                sa[6] += -a[i].y;
                sa[7] += a[i].x;
                
                sa[8] += a[i].x;
                sa[9] += -a[i].y;
                sa[10] += 1;
                sa[11] += 0;
                
                sa[12] += a[i].y;
                sa[13] += a[i].x;
                sa[14] += 0;
                sa[15] += 1;
                
                sb[0] += a[i].x*b[i].x + a[i].y*b[i].y;
                sb[1] += a[i].x*b[i].y - a[i].y*b[i].x;
                sb[2] += b[i].x;
                sb[3] += b[i].y;
            }
            
            cvSolve(&A, &B, &MM, CV_SVD);
            
            om[0] = om[4] = m[0];
            om[1] = -m[1];
            om[3] = m[1];
            om[2] = m[2];
            om[5] = m[3];
        }
    }

    static int EstimateRigidTransform(cv::AutoBuffer<int> &good_idx,
                               int			 	   &good_count,
                               const CvArr* matA, const CvArr* matB, CvMat* matM, int full_affine)
    {
        const int COUNT = 15;
        const int WIDTH = 160, HEIGHT = 120;
        const int RANSAC_MAX_ITERS = 500;
        const int RANSAC_SIZE0 = 3;
        const double RANSAC_GOOD_RATIO = 0.8;
        
        cv::Ptr<CvMat> sA, sB;
        cv::AutoBuffer<CvPoint2D32f> pA, pB;
        cv::AutoBuffer<char> status;
        cv::Ptr<CvMat> gray;
        
        CvMat stubA, *A = cvGetMat(matA, &stubA);
        CvMat stubB, *B = cvGetMat(matB, &stubB);
        CvSize sz0, sz1;
        int cn, equal_sizes;
        int i, j, k, k1;
        int count_x, count_y, count = 0;
        double scale = 1;
        CvRNG rng = cvRNG(-1);
        double m[6] = { 0 };
        CvMat M = cvMat(2, 3, CV_64F, m);
        good_count = 0;
        CvRect brect;
        
        if (!CV_IS_MAT(matM))
            CV_Error(matM ? CV_StsBadArg : CV_StsNullPtr, "Output parameter M is not a valid matrix");
        
        if (!CV_ARE_SIZES_EQ(A, B))
            CV_Error(CV_StsUnmatchedSizes, "Both input images must have the same size");
        
        if (!CV_ARE_TYPES_EQ(A, B))
            CV_Error(CV_StsUnmatchedFormats, "Both input images must have the same data type");
        
        if (CV_MAT_TYPE(A->type) == CV_32FC2 || CV_MAT_TYPE(A->type) == CV_32SC2)
        {
            count = A->cols*A->rows;
            CvMat _pA, _pB;
            pA.allocate(count);
            pB.allocate(count);
            _pA = cvMat(A->rows, A->cols, CV_32FC2, pA);
            _pB = cvMat(B->rows, B->cols, CV_32FC2, pB);
            cvConvert(A, &_pA);
            cvConvert(B, &_pB);
        }
        else
            CV_Error(CV_StsUnsupportedFormat, "Both input images must have either 8uC1 or 8uC3 type");
        
        good_idx.allocate(count);
        
        if (count < RANSAC_SIZE0){
            //printf("too small!\n");
            return 0;
        }
        
        CvMat _pB = cvMat(1, count, CV_32FC2, pB);
        brect = cvBoundingRect(&_pB, 1);
        
        // RANSAC stuff:
        // 1. find the consensus
        for (k = 0; k < RANSAC_MAX_ITERS; k++)
        {
            //printf("ransac %d\n", k);
            int idx[RANSAC_SIZE0];
            CvPoint2D32f a[3];
            CvPoint2D32f b[3];
            
            memset(a, 0, sizeof(a));
            memset(b, 0, sizeof(b));
            
            // choose random 3 non-complanar points from A & B
            for (i = 0; i < RANSAC_SIZE0; i++)
            {
                for (k1 = 0; k1 < RANSAC_MAX_ITERS; k1++)
                {
                    idx[i] = cvRandInt(&rng) % count;
                    
                    for (j = 0; j < i; j++)
                    {
                        if (idx[j] == idx[i])
                            break;
                        // check that the points are not very close one each other
                        if (fabs(pA[idx[i]].x - pA[idx[j]].x) +
                            fabs(pA[idx[i]].y - pA[idx[j]].y) < FLT_EPSILON)
                            break;
                        if (fabs(pB[idx[i]].x - pB[idx[j]].x) +
                            fabs(pB[idx[i]].y - pB[idx[j]].y) < FLT_EPSILON)
                            break;
                    }
                    
                    if (j < i)
                        continue;
                    
                    if (i + 1 == RANSAC_SIZE0)
                    {
                        // additional check for non-complanar vectors
                        a[0] = pA[idx[0]];
                        a[1] = pA[idx[1]];
                        a[2] = pA[idx[2]];
                        
                        b[0] = pB[idx[0]];
                        b[1] = pB[idx[1]];
                        b[2] = pB[idx[2]];
                        
                        double dax1 = a[1].x - a[0].x, day1 = a[1].y - a[0].y;
                        double dax2 = a[2].x - a[0].x, day2 = a[2].y - a[0].y;
                        double dbx1 = b[1].x - b[0].x, dby1 = b[1].y - b[0].y;
                        double dbx2 = b[2].x - b[0].x, dby2 = b[2].y - b[0].y;
                        const double eps = 0.01;
                        
                        if (fabs(dax1*day2 - day1*dax2) < eps*sqrt(dax1*dax1 + day1*day1)*sqrt(dax2*dax2 + day2*day2) ||
                            fabs(dbx1*dby2 - dby1*dbx2) < eps*sqrt(dbx1*dbx1 + dby1*dby1)*sqrt(dbx2*dbx2 + dby2*dby2))
                            continue;
                    }
                    break;
                }
                
                if (k1 >= RANSAC_MAX_ITERS)
                    break;
            }
            
            if (i < RANSAC_SIZE0)
                continue;
            
            // estimate the transformation using 3 points
            icvGetRTMatrix(a, b, 3, &M, full_affine);
            
            for (i = 0, good_count = 0; i < count; i++)
            {
                if (fabs(m[0] * pA[i].x + m[1] * pA[i].y + m[2] - pB[i].x) +
                    fabs(m[3] * pA[i].x + m[4] * pA[i].y + m[5] - pB[i].y) < MAX(brect.width, brect.height)*0.005)
                    good_idx[good_count++] = i;
            }
            
            if (good_count >= count*RANSAC_GOOD_RATIO)
                break;
        }
        
        if (k >= RANSAC_MAX_ITERS){
            //printf("ransac_max_iters\n");
            return 0;
        }
        
        if (good_count < count)
        {
            for (i = 0; i < good_count; i++)
            {
                j = good_idx[i];
                pA[i] = pA[j];
                pB[i] = pB[j];
            }
        }
        
        icvGetRTMatrix(pA, pB, good_count, &M, full_affine);
        m[2] /= scale;
        m[5] /= scale;
        cvConvert(&M, matM);
        
        return 1;
    }

    static cv::Mat EstimateRigidTransform(cv::AutoBuffer<int> &goodIdx,
                                   int &good_count,
                                   cv::InputArray src1,
                                   cv::InputArray src2,
                                   bool fullAffine){
        cv::Mat M(2, 3, CV_64F), A = src1.getMat(), B = src2.getMat();
        CvMat matA = A, matB = B, matM = M;
        int err = EstimateRigidTransform(goodIdx, good_count, &matA, &matB, &matM, fullAffine);
        if (err == 1){
            return M;
        }else{
            M.setTo(0);
            M.at<double>(0, 0) = 1.0;
            M.at<double>(1, 1) = 1.0;
            return M;
        }
    }
    
    static bool homographyRefineLM(const std::vector<cv::Point2d>& pts1,
                            const std::vector<cv::Point2d>& pts2,
                            cv::Mat& homography,const int maxIters){
        
        CvLevMarq solver(8, 0, cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, maxIters, DBL_EPSILON));
        CvMat modelPart = cvMat( solver.param->rows, solver.param->cols,homography.type(),homography.ptr());
        cvCopy( &modelPart, solver.param );
        
        for(;;)
        {
            const CvMat* _param = 0;
            CvMat *_JtJ = 0, *_JtErr = 0;
            double* _errNorm = 0;
            
            if( !solver.updateAlt( _param, _JtJ, _JtErr, _errNorm ))
                break;
            
            for(int i = 0; i <pts1.size();i++ )
            {
                const double* h = _param->data.db;
                double Mx = pts1[i].x, My = pts1[i].y;
                double ww = h[6]*Mx + h[7]*My + 1.;
                ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
                double _xi = (h[0]*Mx + h[1]*My + h[2])*ww;
                double _yi = (h[3]*Mx + h[4]*My + h[5])*ww;
                double err[] = { _xi - pts2[i].x, _yi - pts2[i].y };
                if( _JtJ || _JtErr )
                {
                    double J[][8] =
                    {
                        { Mx*ww, My*ww, ww, 0, 0, 0, -Mx*ww*_xi, -My*ww*_xi },
                        { 0, 0, 0, Mx*ww, My*ww, ww, -Mx*ww*_yi, -My*ww*_yi }
                    };
                    
                    for(int j = 0; j < 8; j++ )
                    {
                        for(int k = j; k < 8; k++ )
                            _JtJ->data.db[j*8+k] += J[0][j]*J[0][k] + J[1][j]*J[1][k];
                        _JtErr->data.db[j] += J[0][j]*err[0] + J[1][j]*err[1];
                    }
                }
                if( _errNorm )
                    *_errNorm += err[0]*err[0] + err[1]*err[1];
            }
        }
        cvCopy(solver.param,&modelPart );
        return true;
    }
}

namespace GSLAM{
static cv::Mat EstimateHomography(
                                  const std::vector<cv::Point2f>& pts1,
                                  const std::vector<cv::Point2f>& pts2,
                                  std::vector<bool>& status,
                                  int& good_count){
    
    cv::AutoBuffer<int> goodIdx;
    bool fullAffine=false;
    cv::Mat rigid=GSLAM::EstimateRigidTransform(goodIdx,good_count,pts1,pts2,fullAffine);
    
    cv::Mat homography=cv::Mat::eye(3,3,CV_64F);
    for (int r=0;r<2;r++) {
        for (int c=0;c<3;c++) {
            homography.at<double>(r,c)=rigid.at<double>(r,c);
        }
    }
    
    std::vector<cv::Point2d> _pts1(good_count),_pts2(good_count);
    for (int i=0;i<good_count;i++) {
        
        _pts1[i].x=pts1[goodIdx[i]].x;
        _pts1[i].y=pts1[goodIdx[i]].y;
        _pts2[i].x=pts2[goodIdx[i]].x;
        _pts2[i].y=pts2[goodIdx[i]].y;
    }
    GSLAM::homographyRefineLM(_pts1,_pts2,homography,20);
    
    status=std::vector<bool>(pts1.size(),false);
    for (int i=0;i<good_count;i++) {
        status[goodIdx[i]]=true;
    }
    
    return homography;
}
}
#endif /* Homography_h */
