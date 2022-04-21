from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf


class BaseModel(metaclass=ABCMeta):
    """Abstract base class for evaluator class."""

    @abstractmethod
    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        raise NotImplementedError()

    @abstractmethod
    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        raise NotImplementedError()

    @abstractmethod
    def create_losses(self) -> None:
        """Create the losses."""
        raise NotImplementedError()

    @abstractmethod
    def add_optimizer(self) -> None:
        """Add the required optimizer to the graph."""
        raise NotImplementedError()


class MFIPS(BaseModel):
    """Matrix Factorization with Inverse Propensity Score (MF-IPS)."""

    def __init__(
        self, num_users: np.array, num_items: np.array, dim: int, lam: float, eta: float
    ) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.eta = eta

        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name="user_ph")
        self.items = tf.placeholder(tf.int32, [None], name="item_ph")
        self.scores = tf.placeholder(tf.float32, [None, 1], name="score_ph")
        self.labels = tf.placeholder(tf.float32, [None, 1], name="label_ph")

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope("embedding_layer"):
            self.user_embeddings = tf.get_variable(
                "user_embeddings",
                shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.item_embeddings = tf.get_variable(
                "item_embeddings",
                shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.i_embed = tf.nn.embedding_lookup(self.item_embeddings, self.items)
            self.user_b = tf.Variable(tf.zeros([self.num_users]), name="user_b")
            self.item_b = tf.Variable(tf.zeros([self.num_items]), name="item_b")
            self.user_bias = tf.nn.embedding_lookup(self.user_b, self.users)
            self.item_bias = tf.nn.embedding_lookup(self.item_b, self.items)
            self.g_bias = tf.get_variable(
                "g_bias", [1], initializer=tf.constant_initializer(0.0)
            )

        with tf.variable_scope("prediction"):
            self.preds = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed), 1)
            self.preds = self.preds + self.user_bias + self.item_bias + self.g_bias
            self.preds = tf.expand_dims(self.preds, 1)

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope("loss"):
            self.mse = tf.reduce_mean(tf.square(self.labels - self.preds))
            local_losses = tf.square(self.labels - self.preds)
            self.ips_mse = tf.reduce_sum(local_losses / self.scores)
            self.ips_mse /= tf.reduce_sum(1.0 / self.scores)
            reg_embeds = tf.nn.l2_loss(self.user_embeddings)
            reg_embeds += tf.nn.l2_loss(self.item_embeddings)
            reg_bias = tf.nn.l2_loss(self.user_b) + tf.nn.l2_loss(self.item_b)
            self.loss = self.ips_mse + self.lam * (reg_embeds + reg_bias)

    def add_optimizer(self) -> None:
        """Add the required optimizer to the graph."""
        with tf.name_scope("optimizer"):
            self.apply_grads = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(
                self.loss
            )


class MFDR(BaseModel):
    """Matrix Factorization with Doubly Robust (MF-DR)."""

    def __init__(
        self, num_users: np.array, num_items: np.array, dim: int, lam: float, eta: float
    ) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.eta = eta

        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name="user_ph")
        self.items = tf.placeholder(tf.int32, [None], name="item_ph")
        self.observes = tf.placeholder(tf.float32, [None, 1], name="obs_ph")
        self.scores = tf.placeholder(tf.float32, [None, 1], name="score_ph")
        self.labels = tf.placeholder(tf.float32, [None, 1], name="label_ph")

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope("embedding_layer"):
            # embeds for final rating prediction
            self.user_embeddings = tf.get_variable(
                "user_embeddings",
                shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.item_embeddings = tf.get_variable(
                "item_embeddings",
                shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.i_embed = tf.nn.embedding_lookup(self.item_embeddings, self.items)
            self.user_b = tf.Variable(tf.zeros([self.num_users]), name="user_b")
            self.item_b = tf.Variable(tf.zeros([self.num_items]), name="item_b")
            self.user_bias = tf.nn.embedding_lookup(self.user_b, self.users)
            self.item_bias = tf.nn.embedding_lookup(self.item_b, self.items)
            self.g_bias = tf.get_variable(
                "g_bias", [1], initializer=tf.constant_initializer(0.0)
            )

            # embeds for imputation model
            self.user_embeddings_imp = tf.get_variable(
                "user_embeddings_imp",
                shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.item_embeddings_imp = tf.get_variable(
                "item_embeddings_imp",
                shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.u_embed_imp = tf.nn.embedding_lookup(
                self.user_embeddings_imp, self.users
            )
            self.i_embed_imp = tf.nn.embedding_lookup(
                self.item_embeddings_imp, self.items
            )

        with tf.variable_scope("prediction"):
            self.preds = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed), 1)
            self.preds = self.preds + self.user_bias + self.item_bias + self.g_bias
            self.preds = tf.expand_dims(self.preds, 1)

        with tf.variable_scope("imptation"):
            self.preds_imp = tf.reduce_sum(
                tf.multiply(self.u_embed_imp, self.i_embed_imp), 1
            )
            self.preds_imp = tf.expand_dims(self.preds_imp, 1)

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope("loss_final_rating_prediction"):
            self.mse = tf.reduce_mean(tf.square(self.labels - self.preds))
            preds_imp = tf.square(tf.stop_gradient(self.preds_imp))
            square_loss = tf.square(self.labels - self.preds)
            self.ips_mse = tf.reduce_sum(square_loss / self.scores)
            self.ips_mse /= tf.reduce_sum(1.0 / self.scores)
            dr_loss = preds_imp + self.observes * (square_loss - preds_imp)
            self.dr_mse = tf.reduce_sum(dr_loss / self.scores)
            self.dr_mse /= tf.reduce_sum(1.0 / self.scores)
            reg_embeds = tf.nn.l2_loss(self.user_embeddings)
            reg_embeds += tf.nn.l2_loss(self.item_embeddings)
            reg_bias = tf.nn.l2_loss(self.user_b) + tf.nn.l2_loss(self.item_b)
            self.loss = self.dr_mse + self.lam * (reg_embeds + reg_bias)

        with tf.name_scope("loss_imputation"):
            preds = tf.stop_gradient(self.preds)
            local_loss_imp = tf.square((self.labels - preds) - self.preds_imp)
            self.loss_e = tf.reduce_sum(local_loss_imp / self.scores)
            self.loss_e /= tf.reduce_sum(1.0 / self.scores)
            reg_embeds = tf.nn.l2_loss(self.user_embeddings_imp)
            reg_embeds += tf.nn.l2_loss(self.item_embeddings_imp)
            self.loss_imp = self.loss_e + self.lam * reg_embeds

    def add_optimizer(self) -> None:
        """Add the required optimizer to the graph."""
        with tf.name_scope("optimizer"):
            self.apply_grads = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(
                self.loss
            )
            self.apply_grads_imp = tf.train.AdamOptimizer(
                learning_rate=self.eta
            ).minimize(self.loss_imp)


class CausE(BaseModel):
    """Causal Embeddings for Recommendation (CausE)."""

    def __init__(
        self,
        num_users: np.array,
        num_items: np.array,
        dim: int,
        lam: float,
        eta: float,
        domain_pen: float,
    ) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.eta = eta
        self.domain_pen = domain_pen

        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name="user_ph")
        self.items = tf.placeholder(tf.int32, [None], name="item_ph")
        self.labels = tf.placeholder(tf.float32, [None, 1], name="label_ph")

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope("embedding_layer"):
            self.user_embeddings_mcar = tf.get_variable(
                "user_embeddings_mcar",
                shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.item_embeddings_mcar = tf.get_variable(
                "item_embeddings_mcar",
                shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.user_embeddings_mnar = tf.get_variable(
                "user_embeddings_mnar",
                shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.item_embeddings_mnar = tf.get_variable(
                "item_embeddings_mnar",
                shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.user_b = tf.Variable(tf.zeros([self.num_users]), name="user_b")
            self.item_b = tf.Variable(tf.zeros([self.num_items]), name="item_b")
            self.user_bias = tf.nn.embedding_lookup(self.user_b, self.users)
            self.item_bias = tf.nn.embedding_lookup(self.item_b, self.items)
            self.g_bias = tf.get_variable(
                "g_bias", [1], initializer=tf.constant_initializer(0.0)
            )

            self.u_embed_mcar = tf.nn.embedding_lookup(
                self.user_embeddings_mcar, self.users
            )
            self.i_embed_mcar = tf.nn.embedding_lookup(
                self.item_embeddings_mcar, self.items
            )
            self.u_embed_mnar = tf.nn.embedding_lookup(
                self.user_embeddings_mnar, self.users
            )
            self.i_embed_mnar = tf.nn.embedding_lookup(
                self.item_embeddings_mnar, self.items
            )

        with tf.variable_scope("prediction_mcar"):
            self.preds_mcar = tf.reduce_sum(
                tf.multiply(self.u_embed_mcar, self.i_embed_mcar), 1
            )
            self.preds_mcar = tf.expand_dims(self.preds_mcar, 1)

        with tf.variable_scope("prediction_mnar"):
            self.preds_mnar = tf.reduce_sum(
                tf.multiply(self.u_embed_mnar, self.i_embed_mnar), 1
            )
            self.preds_mnar = (
                self.preds_mnar + self.user_bias + self.item_bias + self.g_bias
            )
            self.preds_mnar = tf.expand_dims(self.preds_mnar, 1)

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope("loss_mcar"):
            self.mse_mcar = tf.reduce_mean(tf.square(self.labels - self.preds_mcar))
            reg_embeds = tf.nn.l2_loss(self.user_embeddings_mcar)
            reg_embeds += tf.nn.l2_loss(self.item_embeddings_mcar)
            self.loss_mcar = self.mse_mcar + self.lam * reg_embeds

        with tf.name_scope("loss_mnar"):
            self.mse = tf.reduce_mean(tf.square(self.labels - self.preds_mnar))
            reg_embeds = tf.nn.l2_loss(self.user_embeddings_mnar)
            reg_embeds += tf.nn.l2_loss(self.item_embeddings_mnar)
            reg_bias = tf.nn.l2_loss(self.user_b) + tf.nn.l2_loss(self.item_b)
            diff_user_embeddigs = self.user_embeddings_mnar - tf.stop_gradient(
                self.user_embeddings_mcar
            )
            diff_item_embeddigs = self.item_embeddings_mnar - tf.stop_gradient(
                self.item_embeddings_mcar
            )
            reg_embeds_domain = tf.nn.l2_loss(diff_user_embeddigs) + tf.nn.l2_loss(
                diff_item_embeddigs
            )
            self.loss_mnar = self.mse
            self.loss_mnar += self.lam * (reg_embeds + reg_bias)
            self.loss_mnar += self.domain_pen * reg_embeds_domain

    def add_optimizer(self) -> None:
        """Add the required optimizer to the graph."""
        with tf.name_scope("optimizer"):
            self.apply_grads_mnar = tf.train.AdamOptimizer(
                learning_rate=self.eta
            ).minimize(self.loss_mnar)
            self.apply_grads_mcar = tf.train.AdamOptimizer(
                learning_rate=self.eta
            ).minimize(self.loss_mcar)


class DAMF(BaseModel):
    """Domain Adversarial Matrix Factorization (DAMF)."""

    def __init__(
        self,
        num_users: np.array,
        num_items: np.array,
        dim: int,
        eta: float,
        domain_pen: float,
        lam: float,
        lam_pmd: float = 0.1,
    ) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.lam_pmd = lam_pmd
        self.domain_pen = domain_pen
        self.eta = eta

        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name="user_ph")
        self.items = tf.placeholder(tf.int32, [None], name="item_ph")
        self.users_mcar = tf.placeholder(tf.int32, [None], name="user_mcar_ph")
        self.items_mcar = tf.placeholder(tf.int32, [None], name="item_mcar_ph")
        self.labels = tf.placeholder(tf.float32, [None, 1], name="label_ph")

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope("embedding_layer"):
            self.user_embeddings = tf.get_variable(
                "user_embeddings",
                shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.item_embeddings = tf.get_variable(
                "item_embeddings",
                shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.user_embeddings_pmd = tf.get_variable(
                "user_embeddings_pmd",
                shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            self.item_embeddings_pmd = tf.get_variable(
                "item_embeddings_pmd",
                shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
            )

            # for labeled MNAR data
            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.i_embed = tf.nn.embedding_lookup(self.item_embeddings, self.items)
            self.u_embed_pmd = tf.nn.embedding_lookup(
                self.user_embeddings_pmd, self.users
            )
            self.i_embed_pmd = tf.nn.embedding_lookup(
                self.item_embeddings_pmd, self.items
            )
            # for unlabeled MCAR data
            self.u_embed_mcar = tf.nn.embedding_lookup(
                self.user_embeddings, self.users_mcar
            )
            self.i_embed_mcar = tf.nn.embedding_lookup(
                self.item_embeddings, self.items_mcar
            )
            self.u_embed_pmd_mcar = tf.nn.embedding_lookup(
                self.user_embeddings_pmd, self.users_mcar
            )
            self.i_embed_pmd_mcar = tf.nn.embedding_lookup(
                self.item_embeddings_pmd, self.items_mcar
            )

            self.user_b = tf.Variable(tf.zeros([self.num_users]), name="user_b")
            self.item_b = tf.Variable(tf.zeros([self.num_items]), name="item_b")
            self.user_bias = tf.nn.embedding_lookup(self.user_b, self.users)
            self.item_bias = tf.nn.embedding_lookup(self.item_b, self.items)
            self.user_bias_mcar = tf.nn.embedding_lookup(self.user_b, self.users_mcar)
            self.item_bias_mcar = tf.nn.embedding_lookup(self.item_b, self.items_mcar)
            self.g_bias = tf.get_variable(
                "g_bias", [1], initializer=tf.constant_initializer(0.0)
            )

        with tf.variable_scope("final_rating_prediction"):
            # for labeled MNAR data
            self.preds = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed), 1)
            self.preds = self.preds + self.user_bias + self.item_bias + self.g_bias
            self.preds = tf.expand_dims(self.preds, 1)
            # for unlabeled MCAR data
            self.preds_mcar = tf.reduce_sum(
                tf.multiply(self.u_embed_mcar, self.i_embed_mcar), 1
            )
            self.preds_mcar = (
                self.preds_mcar
                + self.user_bias_mcar
                + self.item_bias_mcar
                + self.g_bias
            )
            self.preds_mcar = tf.expand_dims(self.preds_mcar, 1)

        with tf.variable_scope("prediction_for_pmd"):
            # for labeled MNAR data
            self.preds_pmd = tf.reduce_sum(
                tf.multiply(self.u_embed_pmd, self.i_embed_pmd), 1
            )
            self.preds_pmd = tf.expand_dims(self.preds_pmd, 1)
            # for unlabeled MCAR data
            self.preds_pmd_mcar = tf.reduce_sum(
                tf.multiply(self.u_embed_pmd_mcar, self.i_embed_pmd_mcar), 1
            )
            self.preds_pmd_mcar = tf.expand_dims(self.preds_pmd_mcar, 1)

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope("label_loss"):
            self.mse = tf.reduce_mean(tf.square(self.labels - self.preds))
            preds_pmd = tf.stop_gradient(self.preds_pmd)
            preds_pmd_mcar = tf.stop_gradient(self.preds_pmd_mcar)
            pmd = tf.reduce_mean(tf.square(preds_pmd_mcar - self.preds_mcar))
            pmd -= tf.reduce_mean(tf.square(preds_pmd - self.preds))
            self.val_loss = self.mse + pmd
            reg_embeds = tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(
                self.item_embeddings
            )
            reg_bias = tf.nn.l2_loss(self.user_b) + tf.nn.l2_loss(self.item_b)
            self.loss = (
                self.mse + self.domain_pen * pmd + self.lam * (reg_embeds + reg_bias)
            )

        with tf.name_scope("pmd_loss"):
            preds = tf.stop_gradient(self.preds)
            preds_mcar = tf.stop_gradient(self.preds_mcar)
            pmd = tf.reduce_mean(tf.square(self.preds_pmd_mcar - preds_mcar))
            pmd -= tf.reduce_mean(tf.square(self.preds_pmd - preds))
            reg_embeds_pmd = tf.nn.l2_loss(self.user_embeddings_pmd)
            reg_embeds_pmd += tf.nn.l2_loss(self.item_embeddings_pmd)
            self.loss_pmd = -pmd + self.lam_pmd * reg_embeds_pmd

    def add_optimizer(self) -> None:
        """Add the required optimizer to the graph."""
        with tf.name_scope("optimizer"):
            self.apply_grads = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(
                self.loss
            )
            self.apply_grads_pmd = tf.train.AdamOptimizer(
                learning_rate=self.eta
            ).minimize(self.loss_pmd)
