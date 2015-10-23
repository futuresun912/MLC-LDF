import meka.core.M;
import meka.core.StatUtils;
import weka.core.Instances;

import java.util.Arrays;

/**
 * Created by sunlu on 10/20/15.
 * Designed for calculation of normalized mutual information matrix.
 */
public class StatUtilsPro extends StatUtils{

    // return the minimum entropy of Y_j and Y_k.
    // min { H(Y_j), H(Y_k) }
    public static double minH(double[][] P, int j, int k) {

        double H_j, H_k;
        double p_j = P[j][j];
        double p_k = P[k][k];

        H_j = - p_j * Math.log( p_j ) -
                ( 1 - p_j ) * Math.log( 1 - p_j );
        H_k = - p_k * Math.log( p_k ) -
                ( 1 - p_k ) * Math.log( 1 - p_k );

        return H_j < H_k ? H_j : H_k;
//        return ( H_j + H_k ) / 2;   // other kind of normalized MI
    }

    // calculate the normalized mutual information between Y_j and Y_k
    public static double NI(double[][] P, int j, int k) {

        double NI = 0.0;
        double p_j = P[j][j];
        double p_k = P[k][k];
        double p_jk = P[j][k];

        // NI(1;1)
        NI += p_jk * Math.log( p_jk / (p_j * p_k) );
        // NI(1;0)
        if ( p_j != p_jk )  // 0 * log(0) = 0
            NI += ( p_j - p_jk ) * Math.log( (p_j-p_jk) / (p_j*(1-p_k)) );
        // NI(0;1)
        if ( p_k != p_jk )  // 0 * log(0) = 0
            NI += ( p_k - p_jk ) * Math.log( (p_k-p_jk) / ((1-p_j)*p_k) );
        // NI(0;0)
        NI += (1-p_j-p_k + p_jk) * Math.log((1 - p_j - p_k + p_jk) / ((1-p_j)*(1-p_k)) );

        // normalization
        return NI / minH(P, j, k);
    }

    // calculate the normalized mutual information matrix
    public static double[][] NI(double[][] P) {

        int L = P.length;
        double[][] M = new double[L][L];

        for (int j = 0; j < L; j ++)
            for (int k = j+1; k < L; k ++)
                M[j][k] = NI(P, j, k);

        return M;
    }

    // calculate the normalized mutual information
    // NI(Y_j; Y_k) = I(Y_j; Y_k) / min {H(Y_j), H(Y_k)}
    public static double[][] NormMargDep(Instances D) {

        int N = D.numInstances();
        int[][] C = getApproxC(D);
        double[][] P = getP(C, N);
//        System.out.println("Count matrix: \n"+M.toString(C));
//        System.out.println("Probability matrix: \n"+M.toString(P));

        return NI(P);
    }


    // Calcualte the statistics (Imbalance ratio with its mean and variance) of the dataset
    public static double[] CalcIR(Instances D) {

        int L = D.classIndex();
        int N = D.numInstances();
        int[][] countM = StatUtils.getApproxC(D);
        int[] countA = new int[L];        // count of positive instances
        double[] IR = new double[L];      // save the imbalance ratio
        double meanIR, varIR, CVIR;       // mean, variance and Coefficient of Variance of IR
        double[] posRatio = new double[L]; // the ratio of positives over all instances
        double[] factor = new double[L];  // return the IR factor for Wrapper


        // get the count and ratio of positive instances for each label with the max count
        int maxA = countM[0][0];
        for (int j = 0; j < L; j ++) {
            countA[j] = countM[j][j];
            posRatio[j] = (double)countA[j] / (double)N;
            if (maxA < countA[j])
                maxA = countA[j];
        }
        for (int j = 0; j < L; j ++)
            if (countA[j] == 0)
                countA[j] = maxA;
        System.out.println("CA = "+Arrays.toString(countA));
//        System.out.println(maxA);
        System.out.println(Arrays.toString(posRatio));

        // calculate the label-individual Imbalance Ratio array with its mean
        meanIR = 0.0;
        for (int j = 0; j < L; j++) {
            IR[j] = (double) maxA / (double) countA[j];
            meanIR += IR[j];
//            output[j] = IR[j];
        }
        meanIR = meanIR / (double)L;
        System.out.println(Arrays.toString(IR));

        // calculate the variance of IR array and CVIR
        varIR = 0.0;
        for (int j = 0; j < L; j ++)
            varIR += Math.pow((IR[j]-meanIR), 2) / (L-1);
        CVIR = Math.pow(varIR,0.5)/meanIR;

        // compute the IR facter for balancing Wrapper's merit
        for (int j = 0; j < L; j ++) {
            // posRatio
//            double temp = Math.exp(Math.pow(meanIR,1)*CVIR/Math.pow((posRatio[j]+2),2));
//            double temp = meanIR * CVIR / Math.pow(10, posRatio[j]);
            double temp = Math.exp(Math.pow(IR[j] * meanIR, 0.5) * CVIR);
//            double temp = Math.exp(Math.log(IR[j] * meanIR) * CVIR);
            factor[j] = 2 * (temp - 1) / (temp + 1);
        }

//        output[L+1] = varIR;
//        output[L+1] = CVIR;
        System.out.println("meanIR = "+meanIR);
//        System.out.println("varIR = "+varIR);
        System.out.println("CVIR = "+CVIR);

        return factor;
    }


}
