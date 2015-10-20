import meka.core.M;
import meka.core.StatUtils;
import weka.core.Instances;

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
        System.out.println("Count matrix: \n"+M.toString(C));
        System.out.println("Probability matrix: \n"+M.toString(P));

        return NI(P);
    }
}
