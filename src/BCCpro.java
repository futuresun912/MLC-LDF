import meka.classifiers.multilabel.BCC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import mst.Edge;
import mst.EdgeWeightedGraph;
import mst.KruskalMST;
import weka.core.Instances;
import weka.core.Utils;

import java.util.*;
/**
 * Created by sunlu on 10/20/15.
 * BCC method with marginal dependence matrix calculated based on normalized mutual information.
 */
public class BCCpro extends BCC {

    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);

        m_R = new Random(getSeed());
        int L = D.classIndex();
        int d = D.numAttributes()-L;

        double CD[][] = null;
        // CD is the normalized mutual information matrix.
        CD = StatUtilsPro.NormMargDep(D);
        if (getDebug())
            System.out.println("Normalized MI matrix: \n" + M.toString(CD));

//        // output the CD matrix with high precision
//        if (getDebug()) {
//            for (int j = 0; j < L; j++) {
//                for (int k = 0; k < L; k++) {
//                    if (k < j) {
//                        System.out.print(0.00);
//                        System.out.print(" ");
//                    }
//                    if (k == j) {
//                        System.out.print(1.00);
//                        System.out.print(" ");
//                    }
//                    if (k > j) {
//                        System.out.print(CD[j][k]);
//                        System.out.print(" ");
//                    }
//                }
//                System.out.print("\n");
//            }
//        }

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
        int paL[][] = new int[L][0];
        int visted[] = new int[L];
        Arrays.fill(visted,-1);
        visted[root] = 0;
        treeify(root,paM,paL, visted);
        if (getDebug()) {
            for(int i = 0; i < L; i++) {
                System.out.println("pa_"+i+" = "+Arrays.toString(paL[i]));
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
                System.out.println("\t node h_"+j+" : P(y_"+j+" | x_[1:"+d+"], y_"+Arrays.toString(paL[j])+")");
            nodes[j] = new CNode(j, null, paL[j]);
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
                // set as visited
                //paM[root][j] = 0;
            }
        }
        // go through again
        for(int child : children) {
            treeify(child,paM,paL,visited);
        }
    }

}