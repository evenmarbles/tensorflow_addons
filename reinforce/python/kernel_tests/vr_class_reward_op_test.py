import tensorflow as tf

class ReinforceTest(tf.test.TestCase):
    def testVRClassReward(self):
        vrclassreward_module = tf.load_op_library('_reinforce_ops.so')
        with self.test_session():
            result = vrclassreward_module.vr_class_reward([[.5, .3, .1, .1]], [1])
            self.assertAllEqual(result.eval(), (0, [0]))

            result = vrclassreward_module.vr_class_reward([[.5, .3, .1, .1]], [0])
            self.assertAllEqual(result.eval(), (1, [1]))