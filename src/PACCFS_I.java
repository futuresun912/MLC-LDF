/**
 * Created by pipi on 10/14/15.
 */

import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;

/**
 *  PACC with CFS-based feature selection.
 *  pa(y_j) is set as the start set
 */

public class PACCFS_I extends CC {


    private MLFeaSelect mlFeaSelect;

    // Training a PACC
    public void buildClassifier(Instances D) throws Exception {

        testCapabilities(D);
        int L = D.classIndex();

        // Learning of the polytree
        Polytree polytree = new Polytree();
        polytree.setNumFolds(5);
        polytree.setDepMode(false);
        int[][] pa = polytree.polyTree(D, null);
        m_Chain = polytree.getChainOrder();

        // CFS-based feature selection
        mlFeaSelect = new MLFeaSelect(L);
//        Instances[] newD = mlFeaSelect.feaSelect1(D);
        mlFeaSelect.setWrapperCfs(true);
        double emptyIR = 0.0;
        Instances[] newD = new Instances[L];

        // Building the PACC
        nodes = new CNode[L];
        for (int j : m_Chain) {
            newD[j] = mlFeaSelect.feaSelect2(D, j, pa[j], emptyIR);
            nodes[j] = new CNode(j, null, pa[j]);
            nodes[j].build(newD[j], m_Classifier);
        }

        if (getDebug()) {
            System.out.println(A.toString(m_Chain));
            System.out.println(M.toString(pa));
        }
    }

    // Test on a single instance deterministically
    public double[] distributionForInstance(Instance x) throws Exception {

        int L = x.classIndex();
        double[] y = new double[L];

        // Transform the test instance based on selected features
        Instance[] xd = mlFeaSelect.instTransform(x);

        for (int j : m_Chain) {
            y[j] = nodes[j].classify((Instance)xd[j].copy(), y);  // 1 or 0
        }
        return y;
    }


}
