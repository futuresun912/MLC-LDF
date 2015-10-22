
import meka.classifiers.multilabel.BCC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import meka.core.StatUtils;
import mst.Edge;
import mst.EdgeWeightedGraph;
import mst.KruskalMST;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by sunlu on 9/15/15.
 */
public class BCCFS extends BCC {

    private MLFeaSelect mlFeaSelect;

    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);

        m_R = new Random(getSeed());
        int L = D.classIndex();
        int d = D.numAttributes() - L;
        mlFeaSelect = new MLFeaSelect(L);

//        double[][] CD = StatUtils.margDepMatrix(D, "Ibf");
        double[][] CD = StatUtilsPro.NormMargDep(D);

        CD = M.multiply(CD, -1); // because we want a *maximum* spanning tree
        if (getDebug())
            System.out.println("Make a graph ...");
        EdgeWeightedGraph G = new EdgeWeightedGraph((int) L);
        for (int i = 0; i < L; i++) {
            for (int j = i + 1; j < L; j++) {
                Edge e = new Edge(i, j, CD[i][j]);
                G.addEdge(e);
            }
        }
        KruskalMST mst = new KruskalMST(G);
        int paM[][] = new int[L][L];
        for (Edge e : mst.edges()) {
            int j = e.either();
            int k = e.other(j);
            paM[j][k] = 1;
            paM[k][j] = 1;
        }
        if (getDebug()) System.out.println(M.toString(paM));

        int root = getSeed();
        if (getDebug())
            System.out.println("Make a Tree from Root " + root);
        int paL[][] = new int[L][0];
        int visted[] = new int[L];
        Arrays.fill(visted, -1);
        visted[root] = 0;
        treeify(root, paM, paL, visted);
        if (getDebug()) {
            for (int i = 0; i < L; i++) {
                System.out.println("pa_" + i + " = " + Arrays.toString(paL[i]));
            }
        }
        m_Chain = Utils.sort(visted);
        if (getDebug())
            System.out.println("sequence: " + Arrays.toString(m_Chain));

        // First-stage feature selection
        mlFeaSelect.setNumThreads(8);
        Instances[] newD = mlFeaSelect.feaSelect1(D);

        nodes = new CNode[L];
        for (int j : m_Chain) {
            // Second-stage feature selection
            newD[j] = mlFeaSelect.feaSelect2(newD[j], j, paL[j]);
            nodes[j] = new CNode(j, null, paL[j]);
            nodes[j].build(newD[j], m_Classifier);
        }
    }

    public double[] distributionForInstance(Instance x) throws Exception {
        int L = x.classIndex();
        double[] y = new double[L];

        // Transform the test instance
        Instance[] newX = mlFeaSelect.instTransform(x);

        for (int j : m_Chain) {
            y[j] = nodes[j].classify((Instance)newX[j].copy(), y);
        }

        return y;
    }


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