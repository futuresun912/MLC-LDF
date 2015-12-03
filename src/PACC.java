

import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import meka.core.StatUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Collections;

/**
 * Created by sunlu on 9/15/15.
 */
public class PACC extends CC {

    protected int[][] pa;
//    protected int[][] ch;

    // Training a PACC
    public void buildClassifier(Instances D) throws Exception {

        testCapabilities(D);
        int L = D.classIndex();
        Instances[] newD = null;

//        // Preprocessing for MI estimation
//        MLFeaSelect FSforDep = new MLFeaSelect(L);
//        FSforDep.setFilterIG(true);
//        FSforDep.setPercentFeature(0.1);
//        newD = FSforDep.feaSelect1(D);

        // Learning of the polytree
        Polytree polytree = new Polytree();
//        polytree.setNumFolds(3);
//        polytree.setDepMode(true);
        polytree.setPara1(0.2);
        polytree.setPara2(4);

        pa = polytree.polyTree(D, newD);

//        // obtain the children for each node
//        ch = new int[L][];
//        for (int j = 0; j < L; j ++) {
//            ch[j] = new int[]{};
//            for (int k = 0; k < L; k ++) {
//                for (int l : pa[k]) {
//                    if ( l == j )
//                        ch[j] = A.append(ch[j], k);
//                }
//            }
//        }

        m_Chain = polytree.getChainOrder();

        if (getDebug()) {
            System.out.println("m_Chain: \n"+A.toString(m_Chain));
            System.out.println("pa: \n"+M.toString(pa));
        }

//        if (getDebug()) System.exit(0);

        // Building the PACC
        nodes = new CNode[L];
        for (int j : m_Chain) {
            nodes[j] = new CNode(j, null, pa[j]);
            nodes[j].build(D, m_Classifier);
        }
    }


    // ***************************************************************************************
    // SIMPLE EXHAUSTIVE MANNER FOR PREDICTION ***********************************************
    // ***************************************************************************************
    @Override
    public double[] distributionForInstance(Instance xy) throws Exception {

        int L = xy.classIndex();
        double y[];                            // save the optimal assignment
        double y_[];                           // traverse the path on parent nodes in push function
        double[][][] cpt = new double[L][2][]; // conditional probability tables for all nodes
        int paL;                               // save the number of parents of current node
        int powNumJ;                           // the number of 2 to the power of pa[j].length

        // Step 1: calculate the CPT for all nodes (from root(s) to leaf(s))
        for (int j : m_Chain ) {
            y = new double[L];
            paL = pa[j].length;
            powNumJ = (int) Math.pow(2, paL);
            cpt[j][0] = new double[powNumJ];
            cpt[j][1] = new double[powNumJ];
            y_ = new double [paL];

            for (int k = 0; k < powNumJ; k ++) {
                for (int m = 0; m < paL; m ++) {
                    y[pa[j][m]] = y_[m];
                }
                cpt[j][0][k] = nodes[j].distribution((Instance)xy.copy(),y)[0];
                cpt[j][1][k] = 1.0 - cpt[j][0][k];
                if( push(y_, paL-1) ) {
                    break;
                }
            }
        }

        // step 2: Find maximum of joint probability by inquiring on CPTs
        double w = 0.0;
        y_ = new double[L];
        y = new double[L];

        for(int i = 0; i < 1000000; i++) { // limit to 1m

            // obtain the joint probability given xy and y_, output w_
//            double w_ = 0.0;
//            for ( int j : m_Chain ) {
//                if ( pa[j].length == 0 ) {
//                    w_ += ( y_[j] == 0 ? Math.log(cpt[j][0][0]) : Math.log(cpt[j][1][0]) );
//                } else {
//                    int index = 0;
//                    int count = 0;
//                    for ( int k : pa[j] ) {
//                        index += (int)(Math.pow(2, pa[j].length-1-count) * y_[k]);
//                        count ++;
//                    }
//                    w_ += ( y_[j] == 0 ? Math.log(cpt[j][0][index]) : Math.log(cpt[j][1][index]) );
//                }
//            }

            double w_ = 1.0;
            for ( int j : m_Chain ) {
                if ( pa[j].length == 0 ) {
                    w_ *= ( y_[j] == 0 ? cpt[j][0][0] : cpt[j][1][0] );
                } else {
                    int index = 0;
                    int count = 0;
                    for ( int k : pa[j] ) {
                        index += (int)(Math.pow(2, pa[j].length-1-count) * y_[k]);
                        count ++;
                    }
                    w_ *= ( y_[j] == 0 ? cpt[j][0][index] : cpt[j][1][index] );
                }
            }

            if (w_ > w) {
                if (getDebug()) System.out.println("y' = "+Arrays.toString(y_)+", :"+w_);
                y = Arrays.copyOf(y_,y_.length);
                w = w_;
            }

            if (w >= 0.5)
                break;

//            if (w >= Math.log(0.5))
//                break;

            if (push(y_, L-1)) {
                // Done !
                if (getDebug())
                    System.out.println("Tried all "+(i+1)+" combinations.");
                break;
            }
        }

        return y;
    }

    private boolean push(double y[], int j) {
        if (j < 0 ) {
            return true;
        }
        else if (y[j] < 1) {
            y[j]++;
            return false;
        }
        else {
            y[j] = 0.0;
            return push(y,--j);
        }
    }
    // ***************************************************************************************
    // ***************************************************************************************
    // ***************************************************************************************




    // ***************************************************************************************
    // EXHAUSTIVE MANNER *********************************************************************
    // ***************************************************************************************
//    /**
//     * Push - increment y[0] until = K[0], then reset and start with y[0], etc ...
//     * Basically a counter.
//     * @return	True if finished
//     */
//    private static boolean push(double y[], int K[], int j) {
//        if (j >= y.length) {
//            return true;
//        }
//        else if (y[j] < K[j]-1) {
//            y[j]++;
//            return false;
//        }
//        else {
//            y[j] = 0.0;
//            return push(y,K,++j);
//        }
//    }
//
//    /**
//     * GetKs - return [K_1,K_2,...,K_L] where each Y_j \in {1,...,K_j}.
//     * In the multi-label case, K[j] = 2 for all j = 1,...,L.
//     * @param	D	a dataset
//     * @return	an array of the number of values that each label can take
//     */
//    private static int[] getKs(Instances D) {
//        int L = D.classIndex();
//        int K[] = new int[L];
//        for(int k = 0; k < L; k++) {
//            K[k] = D.attribute(k).numValues();
//        }
//        return K;
//    }
//
//    @Override
//    public double[] distributionForInstance(Instance xy) throws Exception {
//
//        int L = xy.classIndex();
//
//        double y[] = new double[L];
//        double w  = 0.0;
//
//		/*
//		 * e.g. K = [3,3,5]
//		 * we push y_[] from [0,0,0] to [2,2,4] over all necessary iterations.
//		 */
//        int K[] = getKs(xy.dataset());
//        if (getDebug())
//            System.out.println("K[] = "+Arrays.toString(K));
//        double y_[] = new double[L];
//
//        for(int i = 0; i < 1000000; i++) { // limit to 1m
//            //System.out.println(""+i+" "+Arrays.toString(y_));
//            double w_  = A.product(super.probabilityForInstance(xy,y_));
//            if (w_ > w) {
//                if (getDebug()) System.out.println("y' = "+Arrays.toString(y_)+", :"+w_);
//                y = Arrays.copyOf(y_,y_.length);
//                w = w_;
//            }
//            if (push(y_,K,0)) {
//                // Done !
//                if (getDebug())
//                    System.out.println("Tried all "+(i+1)+" combinations.");
//                break;
//            }
//        }
//
//        return y;
//    }
    // ***************************************************************************************
    // ***************************************************************************************
    // ***************************************************************************************


}