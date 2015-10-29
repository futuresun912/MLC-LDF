

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


    // Training a PACC
    public void buildClassifier(Instances D) throws Exception {

        testCapabilities(D);
        int L = D.classIndex();

        // Preprocessing for MI estimation
        MLFeaSelect FSforDep = new MLFeaSelect(L);
        FSforDep.setFilterIG(true);
        FSforDep.setPercentFeature(0.5);
        Instances[] newD = FSforDep.feaSelect1(D);

        // Learning of the polytree
        Polytree polytree = new Polytree();
        polytree.setNumFolds(5);

//        polytree.setDepMode(false);
//        int[][] pa = polytree.polyTree(D, null);

        int[][] pa = polytree.polyTree(D, newD);

        m_Chain = polytree.getChainOrder();

        // Building the PACC
        nodes = new CNode[L];
        for (int j : m_Chain) {
            nodes[j] = new CNode(j, null, pa[j]);
            nodes[j].build(D, m_Classifier);
        }

        if (getDebug()) {
            System.out.println(A.toString(m_Chain));
            System.out.println(M.toString(pa));
        }
    }


//    // ***************************************************************************************
//	// THE MAX-SUM ALGORITHM FOR INFERENCE ***************************************************
//	// ***************************************************************************************
//	@Override
//	public double[] distributionForInstance(Instance xy) throws Exception {
//
//		int L = xy.classIndex();
//		double y[] = null;      // save the optimal assignment
//		double ySum[][] = new double[L][2];   // save the probability of current node for y_j = 0 and 1
//		double y_[];                      // traverse the path on parent nodes in push function
//		double[][] msg = new double[L][]; // the message passing upwards (ragged array)
//		double[][] msg0 = new double[L][]; // the message for y_j = 0
//		double[][] msg1 = new double[L][]; // the message for y_j = 1
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
//			parArrayJ = nodes[j].paY.clone();
//			int parLength = parArrayJ.length;
//			powNumJ = (int) Math.pow(2, parArrayJ.length);
//			msg[j] = new double[powNumJ];
//			msg0[j] = new double[powNumJ];
//			msg1[j] = new double[powNumJ];
//			y_ = new double [parLength];
//
//			for (int k = 0; k < powNumJ; k ++) {           // traverse all the possible values of parent nodes (calculate the CPT for node j)
//
//				for (int m = 0; m < parArrayJ.length; m ++) {
//					y[parArrayJ[m]] = y_[m];
//				}
//
//				y[j] = 0;
//				msg0[j][k] = super.probabilityForInstance(xy, y)[j];
//
//				y[j] = 1;
//				msg1[j][k] = super.probabilityForInstance(xy, y)[j];
//
//
//				for ( int pa : parArrayJ ) {
//					y[j] = 0;
//					msg0[j][k] = msg0[j][k] * ySum[pa][(int) y[pa]];
//
//					y[j] = 1;
//					msg1[j][k] = msg1[j][k] * ySum[pa][(int) y[pa]];
//				}
//
//				ySum[j][0] += msg0[j][k];                     // P( y_j=0 | pa(y_j) ) :  Here is just a approximation
//				ySum[j][1] += msg1[j][k];                     // P( y_j=1 | pa(y_j) ) :  Here is just a approximation
//
//				if( push(y_,parLength-1) ) {
//					break;
//				}
//			}
//		}
//
//		y = new double[L];
//
//		for ( int i = L-1; i >= 0; i--)  {          // traverse all the nodes from leaves to roots
//
//			int j = m_Chain[i];
//			ch = new int[]{};
//
//			parArrayJ = nodes[j].paY.clone();
//			powNumJ = (int) Math.pow(2, parArrayJ.length);
//
//			// Step 2: receive all the messages sent by the children of j   (propagates from leaves to roots)
//			for ( int k = 0; k < L; k++ ) {             // find the children of node j
//				if (paMP[k][j] == 1) {
//					ch = A.append(ch, k);
//				}
//			}
//
//			for ( int c : ch ) {                          // receive the messages sent by the children of j
//
//				Arrays.fill(ySum[c], 0);
//
//				parArrayC = nodes[c].paY.clone();
//				powNumC = (int) Math.pow(2, parArrayC.length);
//				int indexJ = 0;
//
//				for ( int index = 0; index < parArrayC.length; index ++ ) {
//					if ( parArrayC[index] == j )
//						indexJ = index;
//				}
//
//				pos = parArrayC.length - indexJ;
//				int step = (int) Math.pow(2, pos);
//				int iniOne = (int) Math.pow(2, pos - 1);
//
//				for (int k = 0; k < powNumC; k += step ) {     // calculate P( y_c = constant | pa(y_c), y_j = 0 )
//					for ( int m = 0; m < Math.pow(2, pos - 1); m ++ ) {
//						ySum[c][0] += msg[c][m+k];
//					}
//				}
//
//				ySum[j][0] = ySum[j][0] * ySum[c][0];
//
//				for (int k = iniOne ; k < powNumC; k += step  ) {     // calculate P( y_c = constant | pa(y_c), y_j = 1 )
//					for ( int m = 0; m < Math.pow(2, pos - 1); m ++ ) {
//						ySum[c][1] += msg[c][m+k];
//					}
//				}
//
//				ySum[j][1] = ySum[j][1] * ySum[c][1];
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





}