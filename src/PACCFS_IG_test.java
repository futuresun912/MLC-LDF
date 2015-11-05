/**
 * Created by sunlu on 11/5/15.
 */
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
 *  PACC with the IG-based first-stage feature selection.
 *  the number of selected features is adjusted by the IR factor.
 */

public class PACCFS_IG_test extends CC {


    private MLFeaSelect mlFeaSelect;

    // Training a PACC
    public void buildClassifier(Instances D) throws Exception {

        testCapabilities(D);
        int L = D.classIndex();

        // Get the IR factor for Wrapper
        double[] IRfactor = StatUtilsPro.CalcIRFactor(D);
        System.out.println(A.toString(IRfactor));

        // First-stage feature selection
        mlFeaSelect = new MLFeaSelect(L);
//        mlFeaSelect.setPercentFeature(0.2);
//        mlFeaSelect.setNumThreads(8);
        mlFeaSelect.setFilterIG(true);
//        mlFeaSelect.setPercentFeature(0.4);
        Instances[] newD = mlFeaSelect.feaSelect1IR(D, IRfactor);

        // Learning of the polytree
        Polytree polytree = new Polytree();
        polytree.setNumFolds(5);
        polytree.setDepMode(false);
        int[][] pa = polytree.polyTree(D, newD);
        m_Chain = polytree.getChainOrder();

        // Building the PACC
        nodes = new CNode[L];
        for (int j : m_Chain) {
//            newD[j] = mlFeaSelect.feaSelect2(newD[j], j, pa[j]);
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
