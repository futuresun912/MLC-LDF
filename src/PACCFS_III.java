import com.sun.deploy.util.ArrayUtil;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;

import java.util.Arrays;

/**
 * Created by sunlu on 9/11/15.
 */
public class PACCFS_III extends CC {


    private MLFeaSelect mlFeaSelect;

    // Training a PACC
    public void buildClassifier(Instances D) throws Exception {

        testCapabilities(D);
        int L = D.classIndex();

        // First-stage feature selection
        mlFeaSelect = new MLFeaSelect(L);
        mlFeaSelect.setFilterIG(true);
        mlFeaSelect.setPercentFeature(0.4);
        Instances[] newD = mlFeaSelect.feaSelect1(D);

        // Learning of the polytree
        Polytree polytree = new Polytree();
        polytree.setNumFolds(5);
        polytree.setDepMode(false);
        int[][] pa = polytree.polyTree(D, newD);
        m_Chain = polytree.getChainOrder();

        if (getDebug()) {
            System.out.println(A.toString(m_Chain));
            System.out.println(M.toString(pa));
        }

        // Building the PACC
        nodes = new CNode[L];
        double emptyIR = 0.0;
        mlFeaSelect.setWrapperCfs(true);
        for (int j : m_Chain) {
//            newD[j] = mlFeaSelect.feaSelect2CFS(newD[j], j, pa[j], emptyIR);
            newD[j] = mlFeaSelect.feaSelect2(newD[j], j, pa[j], emptyIR);
            nodes[j] = new CNode(j, null, pa[j]);
            nodes[j].build(newD[j], m_Classifier);
        }
        System.out.println("********************************" +
                "\n********************************");
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
