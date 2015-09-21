
import meka.classifiers.multilabel.BR;
import meka.core.MLUtils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by sunlu on 9/15/15.
 */
public class BRFS extends BR {

    protected Classifier m_MultiClassifiers[] = null;
    protected Instances m_InstancesTemplates[] = null;
    protected MLFeaSelect mlFeaSelect;

    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);
        int L = D.classIndex();

        m_MultiClassifiers = AbstractClassifier.makeCopies(m_Classifier, L);
        m_InstancesTemplates = new Instances[L];

        // First-stage feature selection
        mlFeaSelect = new MLFeaSelect(L);
        mlFeaSelect.setNumThreads(4);
        Instances[] newD = new Instances[L];
        mlFeaSelect.setPercentFeature(0.3);
        newD = mlFeaSelect.feaSelect1(D);

        for(int j = 0; j < L; j++) {
            int[] pa = new int[]{};

            // Second-stage feature selection
            newD[j] = mlFeaSelect.feaSelect2(newD[j], j, pa);

            // Remove labels except j-th
            Instances D_j = MLUtils.keepAttributesAt(new Instances(newD[j]), new int[]{j}, L);
            D_j.setClassIndex(0);

            //Build the classifier for that class
            m_MultiClassifiers[j].buildClassifier(D_j);
            m_InstancesTemplates[j] = new Instances(D_j, 0);
        }
    }

    @Override
    public double[] distributionForInstance(Instance x) throws Exception {

        int L = x.classIndex();

        double y[] = new double[L];
        Instance[] newX = mlFeaSelect.instTransform(x);

        for (int j = 0; j < L; j++) {
            Instance x_j = (Instance)newX[j].copy();
            x_j.setDataset(null);
            x_j = MLUtils.keepAttributesAt(x_j, new int[]{j}, L);
            x_j.setDataset(m_InstancesTemplates[j]);

            y[j] = m_MultiClassifiers[j].distributionForInstance(x_j)[1];
        }

        return y;
    }

}
