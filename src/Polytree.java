
import com.sun.xml.internal.messaging.saaj.soap.ver1_1.Header1_1Impl;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import mst.Edge;
import mst.EdgeWeightedGraph;
import mst.KruskalMST;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Arrays;

/**
 * Created by sunlu on 9/14/15.
 * This class should implement the following methods:
 * 1. mutualInfoMatrix
 * 2. ploytreeify
 * 3. causalBasin
 * 4. treeify
 * 5. maxSum
 */

public class Polytree {

    private double[][] CD;
    private boolean[] flagCB;
    private int numVisited;
    private boolean[] visited;
    private int L;
    protected int[] chainOrder;
    protected int numFolds = 5;
    protected boolean depMode = true;  // true for conMI; false for margMI

    protected void setNumFolds(int n) throws Exception {
        this.numFolds = n;
    }

    protected int[] getChainOrder() throws Exception {
        return chainOrder;
    }

    protected void setDepMode(boolean mode) throws Exception {
        this.depMode = mode;
    }


    // Build a polytree on the tree
    protected int[][] polyTree(Instances D, Instances[] newD) throws Exception {

        L = (D==null) ? newD[0].classIndex() : D.classIndex();
        numVisited = 0;
        CD = new double[L][L];
        int root = 0;
        int[][] pa = new int[L][0];
        visited = new boolean[L];
        flagCB = new boolean[L];
        Arrays.fill(visited, false);
        Arrays.fill(flagCB, false);

        if (depMode) {
            // Calculate the conditional MI matrix
            if (newD == null)
                CD = conDepMatrix(D);
            if (newD != null)
                CD = conDepMatrix(newD);
        } else {
            // Calculate the marginal normalized MI matrix
            CD = StatUtilsPro.NormMargDep(D);
        }

        // Build the tree skeleton
        int[][] paTree = skeleton(CD);

        // Find the causal basins
        int[][] paPoly = new int[L][L];
        causalBasin(root, paTree, paPoly);

        System.out.println(M.toString(CD));
        System.out.println(M.toString(paTree));
        System.out.println(M.toString(paPoly));

        // If causal basin can't cover all labels, build a directed tree (paTemp)
        int[][] paTemp = new int[L][0];
        root = -1;
        for (int j = 0; j < L; j ++) {
            for (int k = j; k < L; k ++){
                if (paPoly[j][k] == 1) {
                    System.out.println("root: " + j);
                    root = j;
                    Arrays.fill(visited, false);
                    visited[root] = true;
                    treeify(root, paPoly, paTemp);
                    break;
                }
            }
            if (root != -1) break;
        }

        // Save the parents of every node in the polytree (pa)
        for (int j = 0; j < L; j ++) {
            for (int k = j; k < L; k ++) {
                if (paPoly[j][k] == 3)
                    pa[j] = A.append(pa[j], k);
                if (paPoly[j][k] == 2)
                    pa[k] = A.append(pa[k], j);
            }
        }
        for (int j = 0; j < L; j ++) {
            if (pa[j].length < 1 ) {
                for (int v : paTemp[j]) {
                    pa[j] = A.append(pa[j], v);
                    paPoly[j][v] = 3;
                    paPoly[v][j] = 2;
                }
            }
        }

        // Rank the labels in the polytree (rank)
        root = 0;
        int[] rank = new int[L];
        Arrays.fill(rank, 0);
        Arrays.fill(visited, false);
        rankLabel(root, paPoly, rank);
        chainOrder = Utils.sort(rank);

        // Enhance the polytree
        int[] temp = new int[]{};
        for (int j : chainOrder) {
            for (int k : temp) {
                if (paPoly[j][k] != 3) {
//                    pa[j] = A.append(pa[j], k);
                    if (j < k && CD[j][k] > 0.01)
                        pa[j] = A.append(pa[j], k);
                    if (j > k && CD[k][j] > 0.01)
                        pa[j] = A.append(pa[j], k);
                }
            }
            temp = A.append(temp, j);
        }

        System.out.println(M.toString(CD));
        System.out.println(M.toString(paTree));
        System.out.println(M.toString(paPoly));

        return pa;
    }


    // Calculation of conditional dependence between labels given the observation of instances
    protected double[][] conDepMatrix(Instances newData) throws Exception {

        int L = newData.classIndex();
        CNode[][] miNodes = new CNode[L][];
        int[] paNode;
        Classifier model = new Logistic();
        double MI[][] = new double[L][L];

        // n-fold CV for calculation of MI matrix
        for(int i = 0; i < numFolds; i++) {

            Instances[] D_train = new Instances[L];
            Instances[] D_test = new Instances[L];
            for (int j = 0; j < L; j ++) {
                D_train[j] = newData.trainCV(numFolds, i);
                D_test[j] = newData.testCV(numFolds, i);
            }

            // train L*(L+1)/2 logistic classifiers for calculating conditional probability.
            for (int j = 0; j < L; j ++) {
                miNodes[j] = new CNode[L-j];
                paNode = new int[]{};
                for (int k = j; k < L; k ++) {
                    if (k != j)
                        paNode = A.append(paNode, k);
                    miNodes[j][k-j] = new CNode(j, null, paNode);
                    miNodes[j][k-j].build(D_train[j], model);
                    if (k != j)
                        paNode = A.delete(paNode, 0);
                }
            }

            // calculate the conditional mutual information
            for (int j = 0; j < L; j ++)
                for (int k = j + 1; k < L; k ++ )
                    MI[j][k] = conMI(D_test[j], D_test[k], miNodes, j, k);
            MI = addMatrix(MI, MI);
        }
        MI = M.multiply(MI, 1.0 / (double) numFolds);
        return MI;
    }

    // Calculation of conditional dependence between labels given the observation of instances
    protected double[][] conDepMatrix(Instances[] newData) throws Exception {

        int L = newData[0].classIndex();
        CNode[][] miNodes = new CNode[L][];
        int[] paNode;
        Classifier model = new Logistic();
        double MI[][] = new double[L][L];

        // n-fold CV for calculation of MI matrix
        for(int i = 0; i < numFolds; i++) {

            Instances[] D_train = new Instances[L];
            Instances[] D_test = new Instances[L];
            for (int j = 0; j < L; j ++) {
                D_train[j] = newData[j].trainCV(numFolds, i);
                D_test[j] = newData[j].testCV(numFolds, i);
            }

            // train L*(L+1)/2 logistic classifiers for calculating conditional probability.
            for (int j = 0; j < L; j ++) {
                miNodes[j] = new CNode[L-j];
                paNode = new int[]{};
                for (int k = j; k < L; k ++) {
                    if (k != j)
                        paNode = A.append(paNode, k);
                    miNodes[j][k-j] = new CNode(j, null, paNode);
                    miNodes[j][k-j].build(D_train[j], model);
                    if (k != j)
                        paNode = A.delete(paNode, 0);
                }
            }

            // calculate the conditional mutual information
            for (int j = 0; j < L; j ++)
                for (int k = j + 1; k < L; k ++ )
                    MI[j][k] = conMI(D_test[j], D_test[k], miNodes, j, k);
            MI = addMatrix(MI, MI);
        }

        return MI;
//        return MI / numFolds;
    }


    // use the learned classifiers to get conditional probability
    protected double conMI(Instances D_j, Instances D_k, CNode[][] miNodes, int j, int k) throws Exception {

        int L = D_j.classIndex();
        int N = D_j.numInstances();
        double y[] = new double[L];
        double I = 0.0;       		 	 // conditional mutual information for y_j and y_k
        double p_1, p_2;      			 // p( y_j = 1 | x ), p( y_j = 2 | x )
        double p_12[] = {0.0,0.0};       // p_12[0] = p( y_j = 1 | y_k = 0, x ) and p_12[1] = p( y_j = 1 | y_k = 1, x )

        for (int i = 0; i < N; i ++) {

            Arrays.fill(y, 0);
            p_1 = Math.max( miNodes[j][0].distribution((Instance)D_j.instance(i).copy(), y)[1], 0.000001 );                         // p( y_j = 1 | x )
            p_1 = Math.min(p_1, 0.999999);

            Arrays.fill(y, 0);
            p_2 = Math.max( miNodes[k][0].distribution((Instance)D_k.instance(i).copy(), y)[1], 0.000001 );                           // p( y_k = 1 | x )
            p_2 = Math.min(p_2, 0.999999);

            Arrays.fill(y, 0);
            p_12[0] = Math.max( miNodes[j][k-j].distribution((Instance)D_j.instance(i).copy(), y)[1], 0.000001 );     // p( y_j = 1 | y_k = 0, x )
            p_12[0] = Math.min(p_12[0], 0.999999);

            Arrays.fill(y, 0);
            Arrays.fill(y, k, k+1, 1.0);
            p_12[1] = Math.max( miNodes[j][k-j].distribution((Instance)D_j.instance(i).copy(), y)[1], 0.000001 );     // p( y_j = 1 | y_k = 1, x )
            p_12[1] = Math.min(p_12[1], 0.999999);

            I += ( 1 - p_12[0] ) * ( 1 - p_2 ) * Math.log( ( 1 - p_12[0] ) / ( 1 - p_1 ) );     // I( y_j = 0 ; y_k = 0 )
            I += ( 1 - p_12[1] ) * (     p_2 ) * Math.log( ( 1 - p_12[1] ) / ( 1 - p_1 ) );     // I( y_j = 0 ; y_k = 1 )
            I += (     p_12[0] ) * ( 1 - p_2 ) * Math.log( (     p_12[0] ) / (     p_1 ) );     // I( y_j = 1 ; y_k = 0 )
            I += (     p_12[1] ) * (     p_2 ) * Math.log( (     p_12[1] ) / (     p_1 ) );     // I( y_j = 1 ; y_k = 0 )
        }
        I = I / N;
        return I;
    }


//    // use the learned classifiers to get normalized conditional probability
//    protected double conMI(Instances D_j, Instances D_k, CNode[][] miNodes, int j, int k) throws Exception {
//
//        int L = D_j.classIndex();
//        int N = D_j.numInstances();
//        double y[] = new double[L];
//        double I = 0.0;       		 	 // conditional mutual information for y_j and y_k
//        double H_1 = 0.0;
//        double H_2 = 0.0;                 // conditional entropy for Y_j and Y_k
//        double minH;
//        double p_1, p_2;      			 // p( y_j = 1 | x ), p( y_k = 1 | x )
//        double p_12[] = {0.0,0.0};       // p_12[0] = p( y_j = 1 | y_k = 0, x ) and p_12[1] = p( y_j = 1 | y_k = 1, x )
//
//        for (int i = 0; i < N; i ++) {
//
//            Arrays.fill(y, 0);
//            p_1 = Math.max( miNodes[j][0].distribution((Instance)D_j.instance(i).copy(), y)[1], 0.000001 );                         // p( y_j = 1 | x )
//            p_1 = Math.min(p_1, 0.999999);
//
//            Arrays.fill(y, 0);
//            p_2 = Math.max(miNodes[k][0].distribution((Instance) D_k.instance(i).copy(), y)[1], 0.000001);                           // p( y_k = 1 | x )
//            p_2 = Math.min(p_2, 0.999999);
//
//            Arrays.fill(y, 0);
//            p_12[0] = Math.max(miNodes[j][k - j].distribution((Instance) D_j.instance(i).copy(), y)[1], 0.000001);     // p( y_j = 1 | y_k = 0, x )
//            p_12[0] = Math.min(p_12[0], 0.999999);
//
//            Arrays.fill(y, 0);
//            Arrays.fill(y, k, k+1, 1.0);
//            p_12[1] = Math.max( miNodes[j][k-j].distribution((Instance)D_j.instance(i).copy(), y)[1], 0.000001 );     // p( y_j = 1 | y_k = 1, x )
//            p_12[1] = Math.min(p_12[1], 0.999999);
//
//            // calculation of conditional MI
//            I += ( 1 - p_12[0] ) * ( 1 - p_2 ) * Math.log( ( 1 - p_12[0] ) / ( 1 - p_1 ) );     // I( y_j = 0 ; y_k = 0 )
//            I += ( 1 - p_12[1] ) * (     p_2 ) * Math.log( ( 1 - p_12[1] ) / ( 1 - p_1 ) );     // I( y_j = 0 ; y_k = 1 )
//            I += (     p_12[0] ) * ( 1 - p_2 ) * Math.log( (     p_12[0] ) / (     p_1 ) );     // I( y_j = 1 ; y_k = 0 )
//            I += (     p_12[1] ) * (     p_2 ) * Math.log( (     p_12[1] ) / (     p_1 ) );     // I( y_j = 1 ; y_k = 0 )
//
//            // calculation of conditional entropy
//            H_1 -=      p_1  * Math.log(    p_1);
//            H_1 -= (1 - p_1) * Math.log(1 - p_1);
//            H_2 -=      p_2  * Math.log(    p_2);
//            H_2 -= (1 - p_2) * Math.log(1 - p_2);
//         }
//
//        minH = H_1 < H_2 ? H_1 : H_2;
//
//        return I / minH;
//
//    }


    // C = A + B
    protected double[][] addMatrix(double[][] A, double[][] B) {

        double[][] C = new double[A.length][A[0].length];

        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[i].length; j++ )
                C[i][j] = A[i][j] + B[i][j];

        return C;
    }


    // Learning of the tree skeleton (paTree)
    protected int[][] skeleton(double[][] CD) throws Exception {

        CD = M.multiply(CD, -1);
        EdgeWeightedGraph G = new EdgeWeightedGraph(L);
        for(int i = 0; i < L; i ++) {
            for(int j = i+1; j < L; j++) {
                Edge e = new Edge(i, j, CD[i][j]);
                G.addEdge(e);
            }
        }
//        CD = M.multiply(CD, -1);

        KruskalMST mst = new KruskalMST(G);
        int[][] paTree =new int[L][0];
        for (int j = 0; j < L; j ++)
            paTree[j] = new int[]{};
        for(Edge e : mst.edges()) {
            int j = e.either();
            int k = e.other(j);
            paTree[j] = A.append(paTree[j], k);
            paTree[k] = A.append(paTree[k], j);
        }
        return paTree;
    }


    // Find possible causal basins based on the tree skeleton
    // paPoly contains three types of dependence: connected(1), children(2) and parents(3). (check it from rows not columns)
    private void causalBasin(int root, int[][] paTree, int[][] paPoly) throws Exception {

        if ( visited[root] == false ) {
            if ( paTree[root].length == 1 ) {      // 1 if root isn't multi-parent node
                if (paPoly[root][paTree[root][0]] == 0) {
                    paPoly[root][paTree[root][0]] = 1;
                }
                visited[root] = true;
                numVisited ++;
            } else {                              // 2 if root is multi-parent node
                // Calculation the threshold
                double miSum = 0.0;
                double miThreshold = 0.0;
                for (int j : paTree[root]) {
                    if (root < j)
                        miSum += CD[root][j];
                    else
                        miSum += CD[j][root];
                }
                miThreshold = 0.1 * miSum / (double)paTree[root].length;
                System.out.println("node: "+root+"; Threshold: "+miThreshold);

                for (int j : paTree[root]) {
                    for (int k : paTree[root]) {
                        if ( j < k ) {
                            if (CD[j][k] < miThreshold) {
                                paPoly[root][j] = 3;
                                paPoly[root][k] = 3;
                                paPoly[j][root] = 2;
                                paPoly[k][root] = 2;
                                flagCB[root] = true;
                            }
                        }
                    }
                }
                if (flagCB[root] == true) {       // 2.1 if causal basin exists for root
                    for (int j : paTree[root]) {
                        if (paPoly[root][j] != 3) {
                            paPoly[root][j] = 2;
                            paPoly[j][root] = 3;
                            flagCB[j] = true;
                        }
                    }
                    visited[root] = true;
                    numVisited ++;
                } else {                         // 2.2 if causal basin doesn't exist for root
                    for (int j : paTree[root]) {
                        if (paPoly[root][j] == 0)
                            paPoly[root][j] = 1;
                    }

                    visited[root] = true;
                    numVisited ++;
                }
            }
        }

        for (int connectedNode : paTree[root]) {  // Recursively perform on connected nodes
            if (visited[connectedNode] == false)
                causalBasin(connectedNode, paTree, paPoly);
            else if (numVisited == visited.length)
                break;
            else
                continue;
        }
    }


    // Build a directed tree from a root
    private void treeify(int root, int[][] paPoly, int[][] paTemp) throws Exception {

        int children[] = new int[]{};
        for(int j = 0; j < paPoly[root].length; j ++) {
            if (paPoly[root][j] != 0 && visited[j] == false) {
                visited[j] = true;
                children = A.append(children, j);
                if (paPoly[root][j] != 3)
                    paTemp[j] = A.append(paTemp[j], root);
            }
        }
        // go through again
        for(int child : children)
            treeify(child, paPoly, paTemp);
    }


    // Rank the labels for the chain order
    private void rankLabel(int root, int[][] paPoly, int[] rank) throws Exception {

        if (visited[root] == false) {

            int[] parents = new int[]{};
            int[] children = new int[]{};
            for (int j = 0; j < rank.length; j++) {
                if (paPoly[root][j] == 3) {
                    rank[j] = rank[root] - 1;
                    parents = A.append(parents, j);
                }
                if (paPoly[root][j] == 2) {
                    rank[j] = rank[root] + 1;
                    children = A.append(children, j);
                }
            }
            visited[root] = true;

            for (int parent : parents)
                rankLabel(parent, paPoly, rank);
            for (int child : children)
                rankLabel(child, paPoly, rank);
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
