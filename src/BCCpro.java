import meka.classifiers.multilabel.BCC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import mst.Edge;
import mst.EdgeWeightedGraph;
import mst.KruskalMST;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.*;
/**
 * Created by sunlu on 10/20/15.
 * BCC method with marginal dependence matrix calculated based on normalized mutual information.
 */
public class BCCpro extends BCC {

    private int[][] pa;
    private int[][] ch;

    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);

        m_R = new Random(getSeed());
        int L = D.classIndex();
        int d = D.numAttributes()-L;

        // CD is the normalized mutual information matrix.
       double[][] CD = StatUtilsPro.NormMargDep(D);

        if (getDebug())
            System.out.println("Normalized MI matrix: \n" + M.toString(CD));

        CD = M.multiply(CD,-1); // because we want a *maximum* spanning tree
        if (getDebug())
            System.out.println("Make a graph ...");
        EdgeWeightedGraph G = new EdgeWeightedGraph((int)L);
        for(int i = 0; i < L; i++) {
            for(int j = i+1; j < L; j++) {
                Edge e = new Edge(i, j, CD[i][j]);
                G.addEdge(e);
            }
        }

		/*
		 * Run an off-the-shelf MST algorithm to get a MST.
		 */
        if (getDebug())
            System.out.println("Get an MST ...");
        KruskalMST mst = new KruskalMST(G);

		/*
		 * Define graph connections based on the MST.
		 */
        int paM[][] = new int[L][L];
        for (Edge e : mst.edges()) {
            int j = e.either();
            int k = e.other(j);
            paM[j][k] = 1;
            paM[k][j] = 1;
            //StdOut.println(e);
        }
        if (getDebug()) System.out.println(M.toString(paM));

		/*
		 *  Turn the DAG into a Tree with the m_Seed-th node as root
		 */
        int root = getSeed();
        if (getDebug())
            System.out.println("Make a Tree from Root "+root);
        pa = new int[L][0];
        int visted[] = new int[L];
        Arrays.fill(visted,-1);
        visted[root] = 0;
        treeify(root,paM,pa, visted);
        if (getDebug()) {
            for(int i = 0; i < L; i++) {
                System.out.println("pa_"+i+" = "+Arrays.toString(pa[i]));
            }
        }

        // obtain the children for each node
        ch = new int[L][];
        for (int j = 0; j < L; j ++) {
            ch[j] = new int[]{};
            for (int k = 0; k < L; k ++) {
                for (int l : pa[k]) {
                    if ( l == j )
                        ch[j] = A.append(ch[j], k);
                }
            }
        }

        m_Chain = Utils.sort(visted);
        if (getDebug())
            System.out.println("sequence: "+Arrays.toString(m_Chain));
	   /*
		* Bulid a classifier 'tree' based on the Tree
		*/
        if (getDebug()) System.out.println("Build Classifier Tree ...");
        nodes = new CNode[L];
        for(int j : m_Chain) {
            if (getDebug())
                System.out.println("\t node h_"+j+" : P(y_"+j+" | x_[1:"+d+"], y_"+Arrays.toString(pa[j])+")");
            nodes[j] = new CNode(j, null, pa[j]);
            nodes[j].build(D, m_Classifier);
        }

        if (getDebug()) System.out.println(" * DONE * ");

    }


    /**
     * Treeify - make a tree given the structure defined in paM[][], using the root-th node as root.
     */
    private void treeify(int root, int paM[][], int paL[][], int visited[]) {
        int children[] = new int[]{};
        for(int j = 0; j < paM[root].length; j++) {
            if (paM[root][j] == 1) {
                if (visited[j] < 0) {
                    children = A.append(children, j);
                    paL[j] = A.append(paL[j],root);
                    visited[j] = visited[Utils.maxIndex(visited)] + 1;
                }
            }
        }
        // go through again
        for(int child : children) {
            treeify(child,paM,paL,visited);
        }
    }



    // ***************************************************************************************
	// THE MAX-SUM ALGORITHM FOR PREDICTION **************************************************
	// ***************************************************************************************
	@Override
	public double[] distributionForInstance(Instance xy) throws Exception {

		int L = xy.classIndex();
        double y[];                            // save the optimal assignment
        double[][] yMax = new double[L][];     // save the y_j for local maximum
        double msgSum[][];                     // save the sum of log probability for y_j = 0 and 1
		double y_[];                           // traverse the path on parent nodes in push function
		double[][] msg = new double[L][];      // the message passing upwards (ragged array)
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

        // Step 2: receive all the messages sent by the children of j  (from leaf(s) to root(s))
		for ( int i = L-1; i >= 0; i--)  {
			int j = m_Chain[i];
            paL = pa[j].length;
			powNumJ = (int) Math.pow(2, paL);
            msg[j] = new double[powNumJ];
            msgSum = new double[2][powNumJ];
            yMax[j] = new double[powNumJ];

            // initialization of msgSum j's CPT
            for (int k = 0; k < powNumJ; k ++) {
                msgSum[0][k] = Math.log(cpt[j][0][k]);
                msgSum[1][k] = Math.log(cpt[j][1][k]);
            }
            // collect msg from j's children into msgSum
            for ( int c : ch[j] ) {
                for ( int k = 0; k < powNumJ; k ++ ) {
                    msgSum[0][k] += msg[c][0];
                    msgSum[1][k] += msg[c][1];
                }
            }
            // find the local maximum given y_j = 0 or 1
            for ( int k = 0; k < powNumJ; k ++ ) {
                if (msgSum[0][k] >= msgSum[1][k]) {
                    msg[j][k] = msgSum[0][k];
                    yMax[j][k] = 0.0;
                } else {
                    msg[j][k] = msgSum[1][k];
                    yMax[j][k] = 1.0;
                }
            }
        }

        // Step 3: find the y maximizing the joint probability  (from root(s) to leaf(s))
        int indexJ;
        y = new double[L];
        for (  int j : m_Chain ) {
            if ( pa[j].length == 0) {
                 indexJ = 0;
            } else {
                indexJ = (int)y[pa[j][0]];
            }
            y[j] = yMax[j][indexJ];
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




//    // ***************************************************************************************
//	// Exhaustive search *********************************************************************
//	// ***************************************************************************************
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
//    // ***************************************************************************************
//	// ***************************************************************************************
//	// ***************************************************************************************

}
