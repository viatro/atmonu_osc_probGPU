// -*- c++ -*-
//
// GPU versions of mosc functions
//

#include "stdio.h"

// ERROR CHECKING ///////////////////////////////////////////

#define CUDA_ERROR_CHECK // turn this on and off to disable error checking

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line ) {
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line ) {
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

/////////////////////////////////////////////////////////////

#define elec (0)
#define muon (1)
#define tau  (2)
#define re (0)
#define im (1)

typedef enum nu_type {
    data_type,
    nue_type,
    numu_type,
    nutau_type,
    sterile_type,
    unknown_type} NuType;

typedef enum matrix_type {
    standard_type,
    barger_type} MatrixType;

//#define ZERO_CP
static int matrixtype = standard_type;

/* Flag to tell us if we're doing nu_e or nu_sterile matter effects */
static NuType matterFlavor = nue_type;
static float putMix[3][3][2];

/* 2*sqrt(2)*Gfermi in (eV^2-cm^3)/(mole-GeV) - for e<->[mu,tau] */
//static const float tworttwoGf = 1.52588e-4;

__host__
void setMatterFlavor(int flavor) {
    if (flavor == nue_type) matterFlavor = nue_type;
    else if (flavor == sterile_type) matterFlavor = sterile_type;
    else {
        //fprintf(stderr, "setMatterFlavor: flavor=%d", flavor);
        //moscerr("setMatterFlavor: Illegal flavor.");
    }
}

__host__
void setmix_sin(float s12,float s23,float s13,float dcp, float Mix[3][3][2]) {
    float c12,c23,c13,sd,cd;

    if ( s12>1.0f ) s12=1.0f;
    if ( s23>1.0f ) s23=1.0f;
    if ( s13>1.0f ) s13=1.0f;
    if ( cd >1.0f ) cd =1.0f;

    sd = sin( dcp );
    cd = cos( dcp );

    c12 = sqrt(1.0f-s12*s12);
    c23 = sqrt(1.0f-s23*s23);
    c13 = sqrt(1.0f-s13*s13);

    if ( matrixtype == standard_type ) {
        Mix[0][0][re] =  c12*c13;
        Mix[0][0][im] =  0.0f;
        Mix[0][1][re] =  s12*c13;
        Mix[0][1][im] =  0.0f;
        Mix[0][2][re] =  s13*cd;
        Mix[0][2][im] = -s13*sd;
        Mix[1][0][re] = -s12*c23-c12*s23*s13*cd;
        Mix[1][0][im] =         -c12*s23*s13*sd;
        Mix[1][1][re] =  c12*c23-s12*s23*s13*cd;
        Mix[1][1][im] =         -s12*s23*s13*sd;
        Mix[1][2][re] =  s23*c13;
        Mix[1][2][im] =  0.0f;
        Mix[2][0][re] =  s12*s23-c12*c23*s13*cd;
        Mix[2][0][im] =         -c12*c23*s13*sd;
        Mix[2][1][re] = -c12*s23-s12*c23*s13*cd;
        Mix[2][1][im] =         -s12*c23*s13*sd;
        Mix[2][2][re] =  c23*c13;
        Mix[2][2][im] =  0.0f;
    } else {
        Mix[0][0][re] =  c12;
        Mix[0][0][im] =  0.0f;
        Mix[0][1][re] =  s12*c23;
        Mix[0][1][im] =  0.0f;
        Mix[0][2][re] =  s12*s23;
        Mix[0][2][im] =  0.0f;
        Mix[1][0][re] = -s12*c13;
        Mix[1][0][im] =  0.0f;
        Mix[1][1][re] =  c12*c13*c23+s13*s23*cd;
        Mix[1][1][im] =              s13*s23*sd;
        Mix[1][2][re] =  c12*c13*s23-s13*c23*cd;
        Mix[1][2][im] =             -s13*c23*sd;
        Mix[2][0][re] = -s12*s13;
        Mix[2][0][im] =  0.0f;
        Mix[2][1][re] =  c12*s13*c23-c13*s23*cd;
        Mix[2][1][im] =             -c13*s23*sd;
        Mix[2][2][re] =  c12*s13*s23+c13*c23*cd;
        Mix[2][2][im] =              c13*c23*sd;
    }
}

__host__
void setmass(float dms21, float dms23, float dmVacVac[][3]) {
    float delta=5.0e-9;
    float mVac[3];

    mVac[0] = 0.0f;
    mVac[1] = dms21;
    mVac[2] = dms21+dms23;

    /* Break any degeneracies */
    if (dms21==0.0f) mVac[0] -= delta;
    if (dms23==0.0f) mVac[2] += delta;

    dmVacVac[0][0] = dmVacVac[1][1] = dmVacVac[2][2] = 0.0f;
    dmVacVac[0][1] = mVac[0]-mVac[1]; dmVacVac[1][0] = -dmVacVac[0][1];
    dmVacVac[0][2] = mVac[0]-mVac[2]; dmVacVac[2][0] = -dmVacVac[0][2];
    dmVacVac[1][2] = mVac[1]-mVac[2]; dmVacVac[2][1] = -dmVacVac[1][2];
}

/// onwards are for matter effects calcs

__device__
void get_product(float L, float E, float rho,
float Mix[][3][2], float dmMatVac[][3], float dmMatMat[][3],
int antitype, float product[][3][3][2]) {
    float fac=0.0f;
    float twoEHmM[3][3][3][2];
    float tworttwoGf = 1.52588e-4;

    /* (1/2)*(1/(h_bar*c)) in units of GeV/(eV^2-km) */
    /* Reverse the sign of the potential depending on neutrino type */

    //if (matterFlavor == nue_type) {
        /* If we're doing matter effects for electron neutrinos */
        if (antitype<0) fac =  tworttwoGf*E*rho; /* Anti-neutrinos */
        else            fac = -tworttwoGf*E*rho; /* Real-neutrinos */
    //}
    /*else if (matterFlavor == sterile_type) {
        // If we're doing matter effects for sterile neutrinos
        if (antitype<0) fac = -0.5f*tworttwoGf*E*rho; // Anti-neutrinos
        else            fac =  0.5f*tworttwoGf*E*rho; // Real-neutrinos
} */

    /* Calculate the matrix 2EH-M_j */
    for (int n=0; n<3; n++) {
        for (int m=0; m<3; m++) {
#ifndef ZERO_CP
            twoEHmM[n][m][0][re] = -fac*(Mix[0][n][re]*Mix[0][m][re]+Mix[0][n][im]*Mix[0][m][im]);
            twoEHmM[n][m][0][im] = -fac*(Mix[0][n][re]*Mix[0][m][im]-Mix[0][n][im]*Mix[0][m][re]);
            twoEHmM[n][m][1][re] = twoEHmM[n][m][2][re] = twoEHmM[n][m][0][re];
            twoEHmM[n][m][1][im] = twoEHmM[n][m][2][im] = twoEHmM[n][m][0][im];
#else
            twoEHmM[n][m][0][re] = -fac*(Mix[0][n][re]*Mix[0][m][re]);
            twoEHmM[n][m][0][im] = 0 ;
            twoEHmM[n][m][1][re] = twoEHmM[n][m][2][re] = twoEHmM[n][m][0][re];
            twoEHmM[n][m][1][im] = twoEHmM[n][m][2][im] = twoEHmM[n][m][0][im];
#endif
            if (n==m) for (int j=0; j<3; j++)
            twoEHmM[n][m][j][re] -= dmMatVac[j][n];
        }
    }

    /* Calculate the product in eq.(10) of twoEHmM for j!=k */
    //cudaMemset(product, 0, 3*3*3*2*sizeof(float));
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            for (int k=0; k<3; k++) {

#ifndef ZERO_CP
                product[i][j][0][re] += twoEHmM[i][k][1][re]*twoEHmM[k][j][2][re] - twoEHmM[i][k][1][im]*twoEHmM[k][j][2][im];
                product[i][j][0][im] += twoEHmM[i][k][1][re]*twoEHmM[k][j][2][im] + twoEHmM[i][k][1][im]*twoEHmM[k][j][2][re];
                product[i][j][1][re] += twoEHmM[i][k][2][re]*twoEHmM[k][j][0][re] - twoEHmM[i][k][2][im]*twoEHmM[k][j][0][im];
                product[i][j][1][im] += twoEHmM[i][k][2][re]*twoEHmM[k][j][0][im] + twoEHmM[i][k][2][im]*twoEHmM[k][j][0][re];
                product[i][j][2][re] += twoEHmM[i][k][0][re]*twoEHmM[k][j][1][re] - twoEHmM[i][k][0][im]*twoEHmM[k][j][1][im];
                product[i][j][2][im] += twoEHmM[i][k][0][re]*twoEHmM[k][j][1][im] + twoEHmM[i][k][0][im]*twoEHmM[k][j][1][re];
#else
                product[i][j][0][re] += twoEHmM[i][k][1][re]*twoEHmM[k][j][2][re];
                product[i][j][1][re] += twoEHmM[i][k][2][re]*twoEHmM[k][j][0][re];
                product[i][j][2][re] += twoEHmM[i][k][0][re]*twoEHmM[k][j][1][re];
#endif
            }
#ifndef ZERO_CP
            product[i][j][0][re] /= (dmMatMat[0][1]*dmMatMat[0][2]);
            product[i][j][0][im] /= (dmMatMat[0][1]*dmMatMat[0][2]);
            product[i][j][1][re] /= (dmMatMat[1][2]*dmMatMat[1][0]);
            product[i][j][1][im] /= (dmMatMat[1][2]*dmMatMat[1][0]);
            product[i][j][2][re] /= (dmMatMat[2][0]*dmMatMat[2][1]);
            product[i][j][2][im] /= (dmMatMat[2][0]*dmMatMat[2][1]);
#else
            product[i][j][0][re] /= (dmMatMat[0][1]*dmMatMat[0][2]);
            product[i][j][1][re] /= (dmMatMat[1][2]*dmMatMat[1][0]);
            product[i][j][2][re] /= (dmMatMat[2][0]*dmMatMat[2][1]);
#endif
        }
    }
}
/***********************************************************************
getM
Compute the matter-mass vector M, dM = M_i-M_j and
and dMimj. type<0 means anti-neutrinos type>0 means "real" neutrinos
***********************************************************************/

__device__
void getM(float Enu, float rho,
float Mix[][3][2], float dmVacVac[][3], int antitype,
float dmMatMat[][3], float dmMatVac[][3]) {
    int i, j, k;
    float alpha, beta, gamma, fac=0.0f, arg, tmp;
    float alphaV, betaV, gammaV, argV, tmpV;
    float theta0, theta1, theta2;
    float theta0V, theta1V, theta2V;
    float mMatU[3], mMatV[3], mMat[3];
    float tworttwoGf = 1.52588e-4;

    /* Equations (22) fro Barger et.al.*/
    /* Reverse the sign of the potential depending on neutrino type */
    //if (matterFlavor == nue_type) {
    /* If we're doing matter effects for electron neutrinos */
    if (antitype<0) fac =  tworttwoGf*Enu*rho; /* Anti-neutrinos */
    else            fac = -tworttwoGf*Enu*rho; /* Real-neutrinos */
    //}

    //else if (matterFlavor == sterile_type) {
    /* If we're doing matter effects for sterile neutrinos */
    //if (antitype<0) fac = -0.5f*tworttwoGf*Enu*rho; /* Anti-neutrinos */

    //   else        fac =  0.5f*tworttwoGf*Enu*rho; /* Real-neutrinos */
    // }
    /* The strategy to sort out the three roots is to compute the vacuum
    * mass the same way as the "matter" masses are computed then to sort
    * the results according to the input vacuum masses
    */

    alpha  = fac + dmVacVac[0][1] + dmVacVac[0][2];
    alphaV = dmVacVac[0][1] + dmVacVac[0][2];

#ifndef ZERO_CP
    beta = dmVacVac[0][1]*dmVacVac[0][2] + fac*(
        dmVacVac[0][1]*(1.0f - Mix[elec][1][re]*Mix[elec][1][re] - Mix[elec][1][im]*Mix[elec][1][im]) +
        dmVacVac[0][2]*(1.0f - Mix[elec][2][re]*Mix[elec][2][re] - Mix[elec][2][im]*Mix[elec][2][im]));
    betaV = dmVacVac[0][1]*dmVacVac[0][2];
#else
    beta = dmVacVac[0][1]*dmVacVac[0][2] + fac*(
        dmVacVac[0][1]*(1.0f - Mix[elec][1][re]*Mix[elec][1][re]) +
        dmVacVac[0][2]*(1.0f - Mix[elec][2][re]*Mix[elec][2][re]));
    betaV = dmVacVac[0][1]*dmVacVac[0][2];
#endif
#ifndef ZERO_CP
    gamma = fac*dmVacVac[0][1]*dmVacVac[0][2]*(Mix[elec][0][re]*Mix[elec][0][re]+Mix[elec][0][im]*Mix[elec][0][im]);
    gammaV = 0.0f;
#else
    gamma = fac*dmVacVac[0][1]*dmVacVac[0][2]*(Mix[elec][0][re]*Mix[elec][0][re]);
    gammaV = 0.0f;
#endif

    /* Compute the argument of the arc-cosine */
    tmp = alpha*alpha-3.0f*beta;
    tmpV = alphaV*alphaV-3.0f*betaV;
    if (tmp<0.0f) {
        // fprintf(stderr, "getM: alpha^2-3*beta < 0 !\n");
        tmp = 0.0f;
    }

    /* Equation (21) */
    arg = (2.0f*alpha*alpha*alpha-9.0f*alpha*beta+27.0f*gamma)/
    (2.0f*sqrt(tmp*tmp*tmp));
    if (fabs(arg)>1.0f) arg = arg/fabs(arg);
    argV = (2.0f*alphaV*alphaV*alphaV-9.0f*alphaV*betaV+27.0f*gammaV)/
    (2.0f*sqrt(tmpV*tmpV*tmpV));
    if (fabs(argV)>1.0f) argV = argV/fabs(argV);

    /* These are the three roots the paper refers to */
    theta0 = acos(arg)/3.0f;
    theta1 = theta0-(2.0f*M_PI/3.0f);
    theta2 = theta0+(2.0f*M_PI/3.0f);

    theta0V = acos(argV)/3.0f;
    theta1V = theta0V-(2.0f*M_PI/3.0f);
    theta2V = theta0V+(2.0f*M_PI/3.0f);

    mMatU[0] = mMatU[1] = mMatU[2] = -(2.0f/3.0f)*sqrt(tmp);
    mMatU[0] *= cos(theta0);
    mMatU[1] *= cos(theta1);
    mMatU[2] *= cos(theta2);

    tmp = dmVacVac[0][0] - alpha/3.0f;
    mMatU[0] += tmp;
    mMatU[1] += tmp;
    mMatU[2] += tmp;
    mMatV[0] = mMatV[1] = mMatV[2] = -(2.0f/3.0f)*sqrt(tmpV);
    mMatV[0] *= cos(theta0V);
    mMatV[1] *= cos(theta1V);
    mMatV[2] *= cos(theta2V);
    tmpV = dmVacVac[0][0] - alphaV/3.0f;

    mMatV[0] += tmpV;
    mMatV[1] += tmpV;
    mMatV[2] += tmpV;

    /* Sort according to which reproduce the vaccum eigenstates */
    for (i=0; i<3; i++) {
        tmpV = fabs(dmVacVac[i][0]-mMatV[0]);
        k = 0;
        for (j=1; j<3; j++) {
            tmp = fabs(dmVacVac[i][0]-mMatV[j]);
            if (tmp<tmpV) {
                k = j;
                tmpV = tmp;
            }
        }
        mMat[i] = mMatU[k];
    }
    for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
            dmMatMat[i][j] = mMat[i] - mMat[j];
            dmMatVac[i][j] = mMat[i] - dmVacVac[j][0];
        }
    }
}

/***********************************************************************
getA
Calculate the transition amplitude matrix A (equation 10)
***********************************************************************/

__device__
void getA(float L, float E, float rho,
float Mix[][3][2], float dmMatVac[][3], float dmMatMat[][3],
int antitype, float A[3][3][2], float phase_offset) {
    //int n, m, i, j, k;
    float /*fac=0.0f,*/ arg, c, s;

    float X[3][3][2];
    float product[3][3][3][2];
    /* (1/2)*(1/(h_bar*c)) in units of GeV/(eV^2-km) */
    const float LoEfac = 2.534f;

    if ( phase_offset==0.0f ) {
        get_product(L, E, rho, Mix, dmMatVac, dmMatMat, antitype, product);
    }

    /* Make the sum with the exponential factor */
    //cudaMemset(X, 0, 3*3*2*sizeof(float));
    for (int k=0; k<3; k++) {
        arg = -LoEfac*dmMatVac[k][0]*L/E;
        if ( k==2 ) arg += phase_offset ;
        c = cos(arg);
        s = sin(arg);
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
#ifndef ZERO_CP
                X[i][j][re] += c*product[i][j][k][re] - s*product[i][j][k][im];
                X[i][j][im] += c*product[i][j][k][im] + s*product[i][j][k][re];
#else
                X[i][j][re] += c*product[i][j][k][re];
                X[i][j][im] += s*product[i][j][k][re];
#endif
            }
        }
    }
    //  printf("\n testy %f %f ",X[0][0][im], Mix[0][0][im]);
    /* Compute the product with the mixing matrices */
    //cudaMemset(A, 0, 3*3*2*sizeof(float));
    for(int i=0; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
    for(int k = 0; k < 2; ++k)
    A[i][j][k] = 0;

    for (int n=0; n<3; n++) {
        for (int m=0; m<3; m++) {
            for (int i=0; i<3; i++) {
                for (int j=0; j<3; j++) {
#ifndef ZERO_CP
                    A[n][m][re] +=
                        Mix[n][i][re]*X[i][j][re]*Mix[m][j][re] +
                        Mix[n][i][re]*X[i][j][im]*Mix[m][j][im] +
                        Mix[n][i][im]*X[i][j][re]*Mix[m][j][im] -
                        Mix[n][i][im]*X[i][j][im]*Mix[m][j][re];
                    //printf("regret %f %f %f ",Mix[n][i][re], X[i][j][im], Mix[m][j][im]);
                    A[n][m][im] +=
                        Mix[n][i][im]*X[i][j][im]*Mix[m][j][im] +
                        Mix[n][i][im]*X[i][j][re]*Mix[m][j][re] +
                        Mix[n][i][re]*X[i][j][im]*Mix[m][j][re] -
                        Mix[n][i][re]*X[i][j][re]*Mix[m][j][im];
#else
                    A[n][m][re] += Mix[n][i][re]*X[i][j][re]*Mix[m][j][re];
                    A[n][m][im] += Mix[n][i][re]*X[i][j][im]*Mix[m][j][re];
#endif
                    //printf("\n %i %i %i A %f", n, m, re, A[n][m][re]);
                }
            }
        }
    }
}

////#include "mosc.cu"

static float dm[3][3];
static float mix[3][3][2];
//static float Ain[3][3][2];
static float dm21,dm32,s12,s23,s31,cd;

extern /*"C"*/ __host__
float getMixVal(int x, int y, int z) {
    return mix[x][y][z];
}

extern /*"C"*/ __host__
float getT13() {
    return dm[1][1];
}

__host__
void init_mixing_matrix(float dm21f,float dm32f,float s12f,float s23f,float s31f,float cdf) {
    dm21=dm21f ;  dm32=dm32f ;
    s12=s12f   ;  s23=s23f   ; s31=s31f ;
    cd=cdf;
    setMatterFlavor(nue_type);
    setmix_sin(s12,s23,s31,cd,mix);
    setmass(dm21,dm32,dm);
    //  cudaMalloc((void **) &device_array, size);
    //cudaMalloc((void **) &Ain,3*3*2*sizeof(float));
    //Ain[0][0][re] = Ain[1][1][re]		= Ain[2][2][re] = 1.0f;

    //**********
    /*    printf("dm21,dm32   : %f %f \n",dm21,dm32);
    printf("s12,s23,s31 : %f %f %f \n",s12,s23,s31);
    printf("dm  : %f %f %f \n",dm[0][0],dm[0][1],dm[0][2]);
    printf("dm  : %f %f %f \n",dm[1][0],dm[1][1],dm[1][2]);
    printf("dm  : %f %f %f \n",dm[2][0],dm[2][1],dm[2][2]);
    //***********
    //**********
    printf("mix : %f %f %f \n",mix[0][0][0],mix[0][1][0],mix[0][2][0]);
    printf("mix : %f %f %f \n",mix[1][0][0],mix[1][1][0],mix[1][2][0]);
    printf("mix : %f %f %f \n",mix[2][0][0],mix[2][1][0],mix[2][2][0]);
    printf("mix : %f %f %f \n",mix[0][0][1],mix[0][1][1],mix[0][2][1]);
    printf("mix : %f %f %f \n",mix[1][0][1],mix[1][1][1],mix[1][2][1]);
    printf("mix : %f %f %f \n",mix[2][0][1],mix[2][1][1],mix[2][2][1]);*/
    //***********
}

// main kernel
__global__
void get_vacuum_probability(float mix_device[][3][2], int nutype, int beta, float *energy, int n, float path, float *osc_weight, float tdm21, float tdm32) {
    float lovere ;
    float s21, s32, s31, ss21, ss32, ss31 ;
    int ista, iend ;
    float prob[3][3];
    float prob2[3][3][2];

    // index
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    //  if (idx > n) return;

    // make more precise 20081003 rvw
    lovere= 1.26693281*(path)/(energy[idx]);
    s21 = sin(tdm21*lovere);
    s32 = sin(tdm32*lovere);
    s31 = sin((tdm21+tdm32)*lovere) ;
    ss21 = s21*s21 ;
    ss32 = s32*s32 ;
    ss31 = s31*s31 ;

    /* ista = abs(*nutype) - 1 ; */
    for ( ista=0 ; ista<3 ; ista++ ) {
        for ( iend=0 ; iend<2 ; iend++ ) {
            prob[ista][iend]  = mix_device[ista][0][re]*mix_device[iend][0][re]*
            mix_device[ista][1][re]*mix_device[iend][1][re]*ss21;
            prob[ista][iend] += mix_device[ista][1][re]*mix_device[iend][1][re]*
            mix_device[ista][2][re]*mix_device[iend][2][re]*ss32;
            prob[ista][iend] += mix_device[ista][2][re]*mix_device[iend][2][re]*
            mix_device[ista][0][re]*mix_device[iend][0][re]*ss31;
            if ( iend == ista ) {
                prob[ista][iend]  = 1.0f-4.0f*prob[ista][iend];
            } else {
                prob[ista][iend]  = -4.0f*prob[ista][iend];
            }
        }
        prob[ista][2]=1.0f-prob[ista][0]-prob[ista][1];
    }

    nutype = abs(nutype);
    beta = abs(beta);

    //if ( nutype > 0 )
    float ans = prob[nutype-1][beta-1];
    osc_weight[idx] = ans;

    /*  if ( nutype < 0 ) // assuming CPT!!!
    osc_weight[idx] = prob[beta-1][nutype-1];

    osc_weight[idx]= 1.2;*/
}


extern /*"C"*/ __host__
float* GetVacuumProb( int Alpha, int Beta , float *energy_host, int n, float Path ) {
    // alpha -> 1:e 2:mu 3:tau
    // Energy[GeV]
    // Path[km]
    /// simple referes to the fact that in the 3 flavor analysis
    //  the solar mass term is zero

    // create a pointer to device memory
    float *energy_device;

    // specify size of array
    size_t size = n * sizeof(float);

    // CUDA function to allocate memory of size bytes to the address pointed to by device_array
    cudaMalloc((void **) &energy_device, size);

    // copy the array to be squared to the device
    cudaMemcpy(energy_device, energy_host, size, cudaMemcpyHostToDevice);

    float *osc_weights;
    cudaMalloc((void **) &osc_weights, size);

    // copy the mixing matrix to the device
    size_t mixsize = 3*3*2*sizeof(float);

    typedef float mixArray[3][2];
    mixArray *m = (mixArray*)malloc(mixsize);
    memcpy(m, &mix, mixsize);

    //float mix_device[3][3][2];
    mixArray *mix_device;
    //mix[0][0][0] = 1;
    cudaMalloc((void **) &mix_device,mixsize);
    cudaMemcpy(mix_device, mix, mixsize, cudaMemcpyHostToDevice);

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, get_vacuum_probability, 0, n);
    // Round up according to array size
    gridSize = (n + blockSize - 1) / blockSize;

    get_vacuum_probability<<<gridSize, blockSize>>>( mix_device, Alpha, Beta, energy_device, n, Path, osc_weights,  dm21, dm32);

    //cudaThreadSynchronize();

    // copy the results back
    float *osc_weights_host = (float*)malloc(size);
    cudaMemcpy(osc_weights_host, osc_weights, size, cudaMemcpyDeviceToHost);

    cudaFree(energy_device);
    cudaFree(osc_weights);
    cudaFree(mix_device);

    return osc_weights_host;
}

extern /*"C"*/ __host__
void setMNS(float x12, float x13, float x23, float m21, float m23, float Delta,/* float Energy_ ,*/ bool kSquared) {

    float sin12;
    float sin13;
    float sin23;

    if (kSquared) {
        sin12 = sqrt(x12);
        sin13 = sqrt(x13);
        sin23 = sqrt(x23);
    } else {
        sin12 = sqrt(0.5f*(1.0f - sqrt(1.0f - x12)));
        sin13 = sqrt(0.5f*(1.0f - sqrt(1.0f - x13)));
        sin23 = sqrt(0.5f*(1.0f - sqrt(1.0f - x23)));
    }
    init_mixing_matrix(m21, m23, sin12, sin23, sin13, Delta);

}

//////////////////////////////////////////////////////////////////////////////////
// the following functions are DEVICE functions for the matter effects calculation
//////////////////////////////////////////////////////////////////////////////////

__device__
void clear_complex_matrix(float A[][3][2]) {
    //memset(A,0,sizeof(float)*18); // turn into a cuda fucniton
    // cudaMemset((void **) A,0,sizeof(float)*18);
}

__device__
void copy_complex_matrix(float A[][3][2], float B[][3][2]) {
    //memcpy(B,A,sizeof(float)*18);                      // cuda me!
    //cudaMemcpy(B, A, sizeof(float)*18, cudaMemcpyDeviceToDevice);
    B = A;
}

/*
multiply complex 3x3 matrix and 3 vector
    W = A X V
*/

__device__
void multiply_complex_matvec(float A[][3][2], float V[][2],float W[][2]) {

    for(int i=0;i<3;i++) {
        W[i][re] = A[i][0][re]*V[0][re]-A[i][0][im]*V[0][im]+
        A[i][1][re]*V[1][re]-A[i][1][im]*V[1][im]+
        A[i][2][re]*V[2][re]-A[i][2][im]*V[2][im] ;
        W[i][im] = A[i][0][re]*V[0][im]+A[i][0][im]*V[0][re]+
        A[i][1][re]*V[1][im]+A[i][1][im]*V[1][re]+
        A[i][2][re]*V[2][im]+A[i][2][im]*V[2][re] ;
    }
}

// want to output flavor composition of
// pure mass eigenstate, state
__device__
void convert_from_mass_eigenstate( int state, int flavor, float pure[][2], float mix[3][3][2] ) {
    int    i,j,k;
    float mass    [3][2];
    float conj    [3][3][2];
    int    lstate  = state - 1;
    int    factor  = ( flavor > 0 ? -1.0f : 1.0f );

    // need the conjugate for neutrinos but not for
    // anti-neutrinos
    for (i=0; i<3; i++) {
        mass[i][0] = ( lstate == i ? 1.0f : 0.0f );
        mass[i][1] = (                     0.0f );
    }

    for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
            conj[i][j][re] =        mix[i][j][re];
            conj[i][j][im] = factor*mix[i][j][im];
        }
    }
    multiply_complex_matvec(conj, mass, pure);
}



__device__
void get_transition_matrix(int nutypei,float Enuf,float rhof,float Lenf,float Aout[][3][2],float phase_offsetf, float mix[3][3][2], float dm[3][3]) {
    int nutype, make_average ;
    float Enu, rho, Len ;
    float dmMatVac[3][3], dmMatMat[3][3];
    float phase_offset;
    nutype=nutypei;
    Enu=Enuf ;
    rho=rhof ;
    Len=Lenf ;
    phase_offset = phase_offsetf ;
    /*   propagate_mat(Ain,rho,Len,Enu,mix,dm,nutype,Aout);    */
    getM(Enu, rho, mix, dm, nutype, dmMatMat, dmMatVac);
    getA(Len, Enu, rho, mix, dmMatVac, dmMatMat, nutype, Aout,phase_offset);
    //  Aout[0][0][0] =  dm[0][0];
}


// the colonel! (kernel...)
__global__
void propagateLinear(int Alpha, int Beta, float Path, float Density, float ye, float Mix[3][3][2], float dm[3][3], float *Energy, float *osc_w, int n) {
    // here we go
    bool kUseMassEigenstates = false; // quick hack for now

    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        float Probability[3][3];

        int i,j;

        float TransitionMatrix[3][3][2];
        //  float TransitionProduct[3][3][2];
        //  float TransitionTemp[3][3][2];
        float RawInputPsi[3][2];
        float OutputPsi[3][2];

        get_transition_matrix( Alpha,
        Energy[idx],               // in GeV
        Density * ye,   // density * density_convert
        Path,          // in km
        TransitionMatrix,     // Output transition matrix
        0.0f,
        Mix,
        dm);

        //copy_complex_matrix( TransitionMatrix , TransitionProduct );

        for ( i = 0 ; i < 3 ; i++ ) {
            for ( j = 0 ; j < 3 ; j++ ) {
                RawInputPsi[j][0] = 0.0f;
                RawInputPsi[j][1] = 0.0f;
            }

            if( kUseMassEigenstates ) convert_from_mass_eigenstate( i+1, Alpha,  RawInputPsi, Mix );
            else RawInputPsi[i][0] = 1.0f;

            multiply_complex_matvec( TransitionMatrix /*Product*/, RawInputPsi, OutputPsi );

            Probability[i][0] += OutputPsi[0][0] * OutputPsi[0][0] + OutputPsi[0][1]*OutputPsi[0][1];
            Probability[i][1] += OutputPsi[1][0] * OutputPsi[1][0] + OutputPsi[1][1]*OutputPsi[1][1];
            Probability[i][2] += OutputPsi[2][0] * OutputPsi[2][0] + OutputPsi[2][1]*OutputPsi[2][1];
        }

        // now do the part that getprob usually does
        int In = abs( Alpha );
        int Out = abs( Beta );
        osc_w[idx] = Probability[In-1][Out-1];
    }
}

extern /*"C"*/ __host__
void GetProb(int Alpha, int Beta, float Path, float Density, float ye, float *Energy, int n, float *oscw) {
    // copy DM matrix
    size_t dmsize = 3*3*sizeof(float);
    typedef float dmArray[3];
    dmArray *d = (dmArray*)malloc(dmsize);
    memcpy(d, &dm, dmsize);
    dmArray *dm_device;
    cudaMalloc((void **) &dm_device, dmsize);
    cudaMemcpy(dm_device, dm, dmsize, cudaMemcpyHostToDevice);

    // copy mns matrix to device
    size_t mixsize = 3*3*2*sizeof(float);
    typedef float mixArray[3][2];
    mixArray *m = (mixArray*)malloc(mixsize);
    memcpy(m, &mix, mixsize);
    mixArray *mix_device;
    cudaMalloc((void **) &mix_device,mixsize);
    cudaMemcpy(mix_device, mix, mixsize, cudaMemcpyHostToDevice);

    // copy energy array to device
    size_t size = n * sizeof(float);
    float *energy_device = NULL;

    cudaMalloc((void **) &energy_device, size);
    cudaMemcpy(energy_device, Energy, size, cudaMemcpyHostToDevice);

    // allocate output memory space on the device
    float *osc_weights;
    cudaMalloc((void **) &osc_weights, size);

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, propagateLinear, 0, n);
    // Round up according to array size
    gridSize = (n + blockSize - 1) / blockSize;

    //printf("<<< %d | %d, %d >>>\n", minGridSize, gridSize, blockSize);

    propagateLinear<<<gridSize, blockSize>>>(Alpha, Beta, Path, Density, ye, mix_device, dm_device, energy_device, osc_weights, n);
    CudaCheckError();

    // copy the results back
    cudaMemcpy(oscw, osc_weights, size, cudaMemcpyDeviceToHost);

    cudaFree(energy_device);
    cudaFree(osc_weights);
    cudaFree(mix_device);
    cudaFree(dm_device);
    free(m);
    free(d);
}
