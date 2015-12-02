

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
//        polytree.setPara1(0.3);
//        polytree.setPara2(4);

        pa = polytree.polyTree(D, newD);

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





//    // ***************************************************************************************
//	// THE MAX-SUM ALGORITHM FOR PREDICTION **************************************************
//	// ***************************************************************************************
//	@Override
//	public double[] distributionForInstance(Instance xy) throws Exception {
//
//		int L = xy.classIndex();
//        double y[];                           // save the optimal assignment
//        double[][] yTemp = new double[L][];   // save the y_j for local maximum
//		double ySum[][] = new double[L][2];   // save the probability of current node for y_j = 0 and 1
//		double y_[];                          // traverse the path on parent nodes in push function
//		double[][] msg = new double[L][];     // the message passing upwards (ragged array)
//		double[][] msg0 = new double[L][];    // the message for y_j = 0
//		double[][] msg1 = new double[L][];    // the message for y_j = 1
////        double[][] msgmax = new double[L][];  // the message for max_{y_j} P(y_j|pa(y_j))
//		int[] parArrayJ;                    // save the parents of current node j
//		int[] parArrayC;                    // save the parents of the child node c of j
//		int powNumJ;                        // the number of 2 to the power of parArrayJ.length
//		int powNumC;                        // the number of 2 to the power of parArrayC.length
//		int[] ch;                           // the children set of node j
//		int pos;                            // the position of j in the parent set of j's child
//
//		// Step 1: calculate the CPT for current node j (propagates from roots to leaves)
//		for (int j : m_Chain ) {
//
//			y = new double[L];
////			parArrayJ = nodes[j].paY.clone();
//            parArrayJ = pa[j].clone();
//			int parLength = parArrayJ.length;
//			powNumJ = (int) Math.pow(2, parArrayJ.length);
////            yTemp[j] = new double[powNumJ];
//			msg[j] = new double[powNumJ];
//			msg0[j] = new double[powNumJ];
//			msg1[j] = new double[powNumJ];
////            msgmax[j] = new double[powNumJ];
//			y_ = new double [parLength];
//
//			for (int k = 0; k < powNumJ; k ++) {           // traverse all the possible values of parent nodes (calculate the CPT for node j)
//
//				for (int m = 0; m < parArrayJ.length; m ++) {
//					y[parArrayJ[m]] = y_[m];
//				}
//
////				y[j] = 0;
////				msg0[j][k] = super.probabilityForInstance(xy, y)[j];
//                msg0[j][k] = nodes[j].distribution((Instance)xy.copy(),y)[0];
//                msg1[j][k] = 1.0 - msg0[j][k];
//
////				y[j] = 1;
////				msg1[j][k] = super.probabilityForInstance(xy, y)[j];
////                msg1[j][k] = nodes[j].distribution((Instance)xy.copy(),y)[1];
//
////                if (msg0[j][k] >= msg1[j][k]) {
////                    msgmax[j][k] = msg0[j][k];
////                    yTemp[j][k] = 0.0;
////                } else {
////                    msgmax[j][k] = msg1[j][k];
////                    yTemp[j][k] = 1.0;
////                }
//
////				for ( int pa : parArrayJ ) {
////					y[j] = 0;
////					msg0[j][k] = msg0[j][k] * ySum[pa][(int) y[pa]];
////
////					y[j] = 1;
////					msg1[j][k] = msg1[j][k] * ySum[pa][(int) y[pa]];
////				}
////
////				ySum[j][0] += msg0[j][k];                     // P( y_j=0 | pa(y_j) ) :  Here is just a approximation
////				ySum[j][1] += msg1[j][k];                     // P( y_j=1 | pa(y_j) ) :  Here is just a approximation
//
//				if( push(y_,parLength-1) ) {
//					break;
//				}
//			}
//		}
//
//		y = new double[L];
//        // Step 2: receive all the messages sent by the children of j   (propagates from leaves to roots)
//		for ( int i = L-1; i >= 0; i--)  {          // traverse all the nodes from leaves to roots
//
//			int j = m_Chain[i];
//			ch = new int[]{};
//
////			parArrayJ = nodes[j].paY.clone();
//            parArrayJ = pa[j].clone();
//			powNumJ = (int) Math.pow(2, parArrayJ.length);
//
//
////			for ( int k = 0; k < L; k++ ) {             // find the children of node j
////				if (paMP[k][j] == 1) {
////					ch = A.append(ch, k);
////				}
////			}
//            for (int l = 0; l < L; l ++) {
//                for (int k : pa[l]) {
//                    if ( k == j )
//                        ch = A.append(ch, l);
//                }
//
//            }
//
//            if ( ch.length == 0) {
//                for ( int k = 0; k < pa[j].length; k ++ ) {
//                    if (msg0[j][k] >= msg1[j][k]) {
//                        msg[j][k] = msg0[j][k];
//                        yTemp[j][k] = 0.0;
//                    } else {
//                        msg[j][k] = msg1[j][k];
//                        yTemp[j][k] = 1.0;
//                    }
//                }
//            } else {
//                for ( int c : ch ) {
//                    parArrayC = pa[c].clone();
//                    powNumC = (int) Math.pow(2, parArrayC.length);
//                    int indexJ = 0;
//
//                    for ( int index = 0; index < parArrayC.length; index ++ ) {
//                        if ( parArrayC[index] == j )
//                            indexJ = index;
//                    }
//
//                    pos = parArrayC.length - indexJ;
//                    int step = (int) Math.pow(2, pos);
//                    int iniOne = (int) Math.pow(2, pos - 1);
//
//                }
//
//            }
//
////			for ( int c : ch ) {                          // receive the messages sent by the children of j
////
////				Arrays.fill(ySum[c], 0);
////
//////				parArrayC = nodes[c].paY.clone();
////                parArrayC = pa[c].clone();
////				powNumC = (int) Math.pow(2, parArrayC.length);
////				int indexJ = 0;
////
////				for ( int index = 0; index < parArrayC.length; index ++ ) {
////					if ( parArrayC[index] == j )
////						indexJ = index;
////				}
////
////				pos = parArrayC.length - indexJ;
////				int step = (int) Math.pow(2, pos);
////				int iniOne = (int) Math.pow(2, pos - 1);
////
////				for (int k = 0; k < powNumC; k += step ) {     // calculate P( y_c = constant | pa(y_c), y_j = 0 )
////					for ( int m = 0; m < Math.pow(2, pos - 1); m ++ ) {
////						ySum[c][0] += msg[c][m+k];
////					}
////				}
////
////				ySum[j][0] = ySum[j][0] * ySum[c][0];
////
////				for (int k = iniOne ; k < powNumC; k += step  ) {     // calculate P( y_c = constant | pa(y_c), y_j = 1 )
////					for ( int m = 0; m < Math.pow(2, pos - 1); m ++ ) {
////						ySum[c][1] += msg[c][m+k];
////					}
////				}
////
////				ySum[j][1] = ySum[j][1] * ySum[c][1];
//
//			}
//
//			// Step 3: maximize the local probability for j and assign corresponding value to y_j
//			if ( ySum[j][0] < ySum[j][1] )           // if P( y_j = 0 | pa(y_j))  < P( y_j = 1 | pa(y_j) )
//			{
//				y[j] = 1;
//
//				for ( int k = 0; k < powNumJ; k ++ )          // change the message sent from j ( from P( y_j = 0 )  to P( y_j = 1 ) )
//					msg[j][k] = msg1[j][k];
//
//			} else {
//
//				y[j] = 0;
//
//				for ( int k = 0; k < powNumJ; k ++ )          // change the message sent from j ( from P( y_j = 0 )  to P( y_j = 1 ) )
//					msg[j][k] = msg0[j][k];
//			}
//		}
//		return y;
//	}
//
//    private boolean push(double y[], int j) {
//        if (j < 0 ) {
//            return true;
//        }
//        else if (y[j] < 1) {
//            y[j]++;
//            return false;
//        }
//        else {
//            y[j] = 0.0;
//            return push(y,--j);
//        }
//    }
//    // ***************************************************************************************
//	// ***************************************************************************************
//	// ***************************************************************************************





////  // ***************************************************************************************
////	// Exhaustive search *********************************************************************
////	// ***************************************************************************************
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
////  // ***************************************************************************************
////	// ***************************************************************************************
////	// ***************************************************************************************


}