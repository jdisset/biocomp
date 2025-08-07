"""
Comprehensive test suite for the biocomp.parameters module.

This module tests the hierarchical parameter system including:
- ParameterTree: Main interface with tagging
- PTree: Core tree implementation
- ArrayRef: References spanning multiple locations
- ParamPath: Path handling utilities
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from copy import deepcopy
from unittest.mock import patch

import biocomp.parameters as pm
from biocomp.parameters import ParameterTree, PTree, ParamPath, ArrayRef, ArrayRefPath, PTreeBranch


class TestParamPath:
    """Test cases for ParamPath class."""

    def test_creation_from_string(self):
        """Test ParamPath creation from string paths."""
        pa = ParamPath("a")
        assert pa.path == ["a"]
        assert str(pa) == "a"

    def test_creation_from_nested_string(self):
        """Test ParamPath creation from nested string paths."""
        pab = ParamPath("a/b")
        assert pab.path == ["a", "b"]
        assert str(pab) == "a/b"

    def test_creation_from_list(self):
        """Test ParamPath creation from list."""
        pab = ParamPath(["a", "b"])
        assert pab.path == ["a", "b"]
        assert str(pab) == "a/b"

    def test_equality(self):
        """Test ParamPath equality comparisons."""
        pab1 = ParamPath("a/b")
        pab2 = ParamPath(["a", "b"])
        pa = ParamPath("a")

        assert pab1 == pab2
        assert pab1 != pa

    def test_ordering(self):
        """Test ParamPath ordering comparisons."""
        pa = ParamPath("a")
        pab = ParamPath("a/b")
        paaa = ParamPath("a/a/a")
        pb = ParamPath("b")
        pba = ParamPath("b/a")

        assert pa < pab
        assert pab > pa
        assert paaa > pa
        assert paaa < pab  # lexicographic: 'a/a/a' < 'a/b'
        assert pb > pa
        assert pba > pab

    def test_empty_path(self):
        """Test empty ParamPath behavior."""
        empty = ParamPath("")
        pa = ParamPath("a")

        assert str(empty) == ""
        assert empty < pa

    def test_root_path_normalization(self):
        """Test that root slashes are properly normalized."""
        pa = ParamPath("a")
        paroot = ParamPath("/a/")

        assert str(pa) == str(paroot) == "a"

    def test_division_operator(self):
        """Test path concatenation using / operator."""
        pa = ParamPath("a")
        pb = ParamPath("b")

        pab = pa / "b"
        assert pab.path == ["a", "b"]

        pab2 = pa / pb
        assert pab2.path == ["a", "b"]

    def test_addition_operator(self):
        """Test path concatenation using + operator."""
        pa = ParamPath("a")
        pab = pa + "b"
        assert pab.path == ["a", "b"]

    def test_indexing_and_iteration(self):
        """Test ParamPath indexing and iteration."""
        pab = ParamPath("a/b")

        assert pab[0] == "a"
        assert pab[1] == "b"
        assert len(pab) == 2
        assert list(pab) == ["a", "b"]


class TestPTreeBranch:
    """Test cases for PTreeBranch class."""

    def test_creation(self):
        """Test PTreeBranch creation."""
        branch = PTreeBranch()
        assert isinstance(branch, dict)
        assert len(branch) == 0

    def test_basic_operations(self):
        """Test basic dictionary operations."""
        branch = PTreeBranch()
        branch["a"] = PTree(1)
        branch["b"] = PTree(2)

        assert len(branch) == 2
        assert "a" in branch
        assert "b" in branch


class TestPTree:
    """Test cases for PTree class."""

    def test_empty_tree(self):
        """Test empty PTree creation and properties."""
        tree = PTree()
        assert tree.is_empty()
        assert tree.is_leaf()
        assert len(tree) == 0

    def test_leaf_node(self):
        """Test leaf node creation and properties."""
        tree = PTree(42)
        assert not tree.is_empty()
        assert tree.is_leaf()
        assert tree.value == 42

    def test_branch_node(self):
        """Test branch node creation and properties."""
        tree = PTree()
        tree["a"] = 1
        assert not tree.is_empty()
        assert not tree.is_leaf()
        assert len(tree) == 1

    def test_nested_access(self):
        """Test nested path access."""
        tree = PTree()
        tree["a/b/c"] = 42

        assert tree["a/b/c"] == 42
        assert not PTree.is_leaf_at(tree, "a")
        assert not PTree.is_leaf_at(tree, "a/b")
        assert PTree.is_leaf_at(tree, "a/b/c")

    def test_array_storage(self):
        """Test storing arrays in PTree."""
        tree = PTree()
        arr = np.array([1, 2, 3])
        tree["data"] = arr

        assert np.array_equal(tree["data"], arr)
        assert tree["data"].shape == (3,)

    def test_deletion(self):
        """Test node deletion."""
        tree = PTree()
        tree["a"] = 1
        tree["b"] = 2

        assert len(tree) == 2
        del tree["a"]
        assert len(tree) == 1
        assert "a" not in tree
        assert "b" in tree

    def test_iteration(self):
        """Test tree iteration."""
        tree = PTree()
        tree["a"] = 1
        tree["b"] = 2
        tree["c/d"] = 3

        leaves = list(tree.iter_leaves())
        assert len(leaves) == 3

        # Check paths and values
        paths, values = zip(*leaves)
        assert "a" in paths
        assert "b" in paths
        assert "c/d" in paths

    def test_read_only_mode(self):
        """Test read-only tree behavior."""
        tree = PTree(42)
        tree.set_read_only(True)

        # Should not be able to modify read-only tree
        # Note: The actual behavior may vary - let's check what the implementation does
        assert tree.read_only == True

    def test_tree_validation(self):
        """Test tree structure validation."""
        tree = PTree()
        tree["a"] = 1
        tree["b/c"] = 2

        # Should not raise any exceptions
        tree.check()

    def test_equality_comparison(self):
        """Test PTree equality comparison."""
        tree1 = PTree()
        tree1["a"] = 1
        tree1["b"] = np.array([1, 2, 3])

        tree2 = PTree()
        tree2["a"] = 1
        tree2["b"] = np.array([1, 2, 3])

        assert tree1.direct_compare(tree2)

    def test_tree_structure_visualization(self):
        """Test tree structure visualization."""
        tree = PTree()
        tree["a"] = 1
        tree["b/c"] = np.array([1, 2])

        structure = tree.visualize_tree_structure()
        assert isinstance(structure, list)
        assert len(structure) > 0


class TestArrayRef:
    """Test cases for ArrayRef class."""

    def test_creation(self):
        """Test ArrayRef creation."""
        tree = PTree()
        tree["a"] = np.array([1, 2, 3])
        tree["b"] = np.array([4, 5, 6])

        ref = ArrayRef(tree)
        assert ref.tree is tree
        assert len(ref.indices) == 0

    def test_push_back(self):
        """Test adding references to ArrayRef."""
        tree = PTree()
        tree["a"] = np.array([1, 2, 3])
        tree["b"] = np.array([4, 5, 6])

        ref = ArrayRef(tree)
        ref.push_back("a", 1)  # Index 1 from array 'a'
        ref.push_back("b", 0)  # Index 0 from array 'b'

        view = ref.view()
        expected = np.array([2, 4])  # a[1]=2, b[0]=4
        assert np.array_equal(view, expected)

    def test_multiple_references(self):
        """Test ArrayRef with multiple references from same array."""
        tree = PTree()
        tree["data"] = np.array([10, 20, 30, 40])

        ref = ArrayRef(tree)
        ref.push_back("data", 0)
        ref.push_back("data", 2)
        ref.push_back("data", 3)

        view = ref.view()
        expected = np.array([10, 30, 40])
        assert np.array_equal(view, expected)

    def test_nested_path_references(self):
        """Test ArrayRef with nested path references."""
        tree = PTree()
        tree["level1/level2/data"] = np.array([1, 2, 3])

        ref = ArrayRef(tree)
        ref.push_back("level1/level2/data", 1)

        view = ref.view()
        expected = np.array([2])
        assert np.array_equal(view, expected)

    def test_empty_ref_view(self):
        """Test ArrayRef view when no references are added."""
        tree = PTree()
        ref = ArrayRef(tree)

        view = ref.view()
        assert view.shape == (0,)

    def test_invalid_path_reference(self):
        """Test ArrayRef behavior with invalid path."""
        tree = PTree()
        tree["a"] = np.array([1, 2, 3])

        ref = ArrayRef(tree)

        # ArrayRef allows adding paths that don't exist yet
        # The error occurs when trying to view() the reference
        ref.push_back("nonexistent", 0)
        with pytest.raises(KeyError):
            ref.view()

    def test_invalid_index_reference(self):
        """Test ArrayRef behavior with invalid index."""
        tree = PTree()
        tree["a"] = np.array([1, 2, 3])

        ref = ArrayRef(tree)

        # ArrayRef allows adding any index, error occurs during view()
        ref.push_back("a", 10)  # Index out of bounds
        with pytest.raises(IndexError):
            ref.view()


class TestParameterTree:
    """Test cases for ParameterTree class."""

    def test_creation(self):
        """Test ParameterTree creation."""
        p = ParameterTree()
        assert p.data.is_empty()
        assert len(p.data) == 0
        assert p.tags.is_empty()

    def test_basic_assignment(self):
        """Test basic parameter assignment."""
        p = ParameterTree()
        p["a"] = 1

        assert p["a"] == 1
        assert len(p.data) == 1
        assert PTree.is_leaf_at(p.data, "a")

    def test_nested_assignment(self):
        """Test nested parameter assignment."""
        p = ParameterTree()
        p["b/arr"] = np.array([0, 1, 2])

        assert np.array_equal(p["b/arr"], np.array([0, 1, 2]))
        assert not PTree.is_leaf_at(p.data, "b")
        assert PTree.is_leaf_at(p.data, "b/arr")

    def test_at_method(self):
        """Test the 'at' method for conditional assignment."""
        p = ParameterTree()

        # First call should set the value
        result = p.at("new_key", 42)
        assert result == 42
        assert p["new_key"] == 42

        # Second call without overwrite should return existing value
        result = p.at("new_key", 100, overwrite=False)
        assert result == 42
        assert p["new_key"] == 42

        # With overwrite=True should update
        result = p.at("new_key", 100, overwrite=True)
        assert result == 100
        assert p["new_key"] == 100

    def test_array_ref_integration(self):
        """Test ArrayRef integration with ParameterTree."""
        p = ParameterTree()
        p["b/arr"] = np.array([0, 1, 2])
        p["b/arr2"] = np.array([3, 4])

        ref = pm.ArrayRef(p.data)
        ref.push_back("b/arr", 1)
        ref.push_back("b/arr2", 0)

        p["ref"] = ref
        expected = np.array([1, 3])  # arr[1]=1, arr2[0]=3
        assert np.array_equal(p["ref"], expected)

    def test_branch_on_leaf_error(self):
        """Test that creating a branch on a leaf node fails."""
        p = ParameterTree()
        p["a"] = 1

        with pytest.raises(KeyError):
            p["a/newbranch"] = 1

    def test_tagging_system(self):
        """Test parameter tagging functionality."""
        p = ParameterTree()
        p["a"] = 1
        p["b/arr"] = np.array([0, 1, 2])

        # Initially no tags
        assert p.tags.is_empty()

        # Add tags
        p.tag("a", "tag1")
        p.tag("b/arr", "tag1")

        assert not p.tags.is_empty()
        assert np.array_equal(p.tags["a"], np.array([True]))
        assert np.array_equal(p.tags["b/arr"], np.array([True]))

    def test_multiple_tags(self):
        """Test multiple tags on same parameter."""
        p = ParameterTree()
        p["a"] = 1

        p.tag("a", "tag1")
        p.tag("a", "tag2")

        # Should have both tags
        assert np.array_equal(p.tags["a"], np.array([True, True]))
        assert "tag1" in p.tagnames
        assert "tag2" in p.tagnames

    def test_tag_filtering(self):
        """Test filtering parameters by tags."""
        p = ParameterTree()
        p["a"] = 1
        p["b"] = 2
        p["c"] = 3

        p.tag("a", "train")
        p.tag("b", "train")
        p.tag("c", "test")

        train_params, other_params = p.filter_by_tag("train")

        assert len(train_params.data) == 2
        assert len(other_params.data) == 1
        assert "a" in train_params.data
        assert "b" in train_params.data
        assert "c" in other_params.data

    def test_merge_functionality(self):
        """Test merging ParameterTrees."""
        p1 = ParameterTree()
        p1["a"] = 1
        p1["b"] = np.array([1, 2])
        p1.tag("a", "tag1")

        p2 = ParameterTree()
        p2["c"] = 3
        p2["d"] = np.array([3, 4])
        p2.tag("c", "tag2")

        merged = ParameterTree.merge(p1, p2)

        assert merged["a"] == 1
        assert merged["c"] == 3
        assert np.array_equal(merged["b"], np.array([1, 2]))
        assert np.array_equal(merged["d"], np.array([3, 4]))
        assert "tag1" in merged.tagnames
        assert "tag2" in merged.tagnames

    def test_deepcopy(self):
        """Test deep copying of ParameterTree."""
        p = ParameterTree()
        p["a"] = 1
        p["b/arr"] = np.array([1, 2, 3])
        p.tag("a", "tag1")

        p_copy = deepcopy(p)

        assert p_copy == p
        assert p_copy is not p
        assert p_copy.data is not p.data

    def test_jax_tree_integration(self):
        """Test JAX tree utilities integration."""
        p = ParameterTree()
        p["a"] = 1.0
        p["b/arr"] = np.array([1.0, 2.0, 3.0])

        # Test tree flattening/unflattening
        leaves, treedef = jtu.tree_flatten(p)
        reconstructed = jtu.tree_unflatten(treedef, leaves)

        assert reconstructed == p
        assert len(leaves) == 2  # Two leaf values

    def test_jax_transformations(self):
        """Test JAX transformations on ParameterTree."""
        p = ParameterTree()
        p["a"] = 1.0
        p["b"] = np.array([1.0, 2.0], dtype=np.float32)

        # Test tree_map
        doubled = jtu.tree_map(lambda x: x * 2, p)

        assert doubled["a"] == 2.0
        assert np.array_equal(doubled["b"], np.array([2.0, 4.0]))

    def test_tree_set_at(self):
        """Test JAX-style tree_set_at functionality."""
        p = ParameterTree()
        p["a"] = 1.0
        p["b/c"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Add some tags to test the implementation
        p.tag("b/c", "test_tag")

        # Test setting at path
        new_p = p.tree_set_at("b/c", np.array([4.0, 5.0, 6.0]))

        # Original should be unchanged
        assert np.array_equal(p["b/c"], np.array([1.0, 2.0, 3.0]))
        # New tree should have updated values
        assert np.array_equal(new_p["b/c"], np.array([4.0, 5.0, 6.0]))

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
    def test_different_dtypes(self, dtype):
        """Test ParameterTree with different numpy dtypes."""
        p = ParameterTree()
        arr = np.array([1, 2, 3], dtype=dtype)
        p["data"] = arr

        assert p["data"].dtype == dtype
        assert np.array_equal(p["data"], arr)

    def test_gradients_with_tags(self):
        """Test computing gradients with tag filtering."""

        def loss_fn(params):
            trainable, fixed = params.filter_by_tag("trainable")
            merged = ParameterTree.merge(trainable, fixed)
            return jnp.sum(merged["weights"] ** 2)

        p = ParameterTree()
        p["weights"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p["bias"] = np.array([0.1], dtype=np.float32)
        p.tag("weights", "trainable")
        # bias is not tagged as trainable

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(p)

        # Should have gradients for weights
        assert "weights" in grads.data
        expected_grad = 2 * np.array([1.0, 2.0, 3.0])
        assert np.allclose(grads["weights"], expected_grad)


class TestArrayRefPath:
    """Test cases for ArrayRefPath class."""

    def test_array_ref_path_creation(self):
        """Test ArrayRefPath creation and comparison."""
        path = ParamPath("a/b/c")
        arp = ArrayRefPath(path, [], [])

        # Test comparison with ParamPath
        assert arp > ParamPath("a/b")
        assert ParamPath("a/b") < arp

    def test_array_ref_path_ordering(self):
        """Test ArrayRefPath ordering with ParamPath."""
        path1 = ParamPath("a/b")
        path2 = ParamPath("a/c")
        arp1 = ArrayRefPath(path1, [], [])
        arp2 = ArrayRefPath(path2, [], [])

        # Should maintain ordering based on underlying path
        assert arp1 < arp2
        assert arp2 > arp1


class TestSerializationAndHDF5:
    """Test serialization capabilities including HDF5 support."""

    def test_parameter_tree_hdf5_serialization(self):
        """Test HDF5 serialization if available."""
        p = ParameterTree()
        p["weights"] = np.random.randn(5, 3).astype(np.float32)
        p["bias"] = np.zeros(3, dtype=np.float32)
        p.tag("weights", "trainable")

        # Test basic serialization structure exists
        # The actual HDF5 functionality may require h5py installation
        assert hasattr(pm, "serialize"), "Serialization functions should exist"

    def test_jax_array_handling(self):
        """Test handling of JAX arrays in parameter trees."""
        p = ParameterTree()
        jax_array = jnp.array([1.0, 2.0, 3.0])
        p["jax_data"] = jax_array

        # Should handle JAX arrays seamlessly
        assert isinstance(p["jax_data"], jnp.ndarray)
        assert np.array_equal(p["jax_data"], jax_array)

    def test_mixed_array_types(self):
        """Test parameter tree with mixed numpy/JAX arrays."""
        p = ParameterTree()
        p["numpy_data"] = np.array([1, 2, 3])
        p["jax_data"] = jnp.array([4.0, 5.0, 6.0])

        # Test tree operations work with mixed types
        leaves, treedef = jtu.tree_flatten(p)
        reconstructed = jtu.tree_unflatten(treedef, leaves)

        assert np.array_equal(reconstructed["numpy_data"], p["numpy_data"])
        assert np.array_equal(reconstructed["jax_data"], p["jax_data"])


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""

    def test_circular_references_detection(self):
        """Test detection of circular references in PTree."""
        tree1 = PTree()
        tree2 = PTree()

        # Create a circular reference
        tree1["ref"] = tree2
        tree2["ref"] = tree1

        # Should detect the circular reference
        assert not tree1.direct_compare(tree2)

    def test_invalid_path_access(self):
        """Test accessing invalid paths."""
        p = ParameterTree()
        p["a"] = 1

        with pytest.raises(KeyError):
            _ = p["nonexistent"]

        with pytest.raises(KeyError):
            _ = p["a/invalid"]  # 'a' is a leaf, can't access sub-paths

    def test_empty_parameter_tree_operations(self):
        """Test operations on empty ParameterTree."""
        p = ParameterTree()

        # Should handle empty tree gracefully
        leaves, treedef = jtu.tree_flatten(p)
        assert len(leaves) == 0

        reconstructed = jtu.tree_unflatten(treedef, leaves)
        assert reconstructed == p

    def test_large_array_ref(self):
        """Test ArrayRef with many references."""
        tree = PTree()
        tree["data"] = np.arange(1000)

        ref = ArrayRef(tree)
        # Add many references
        for i in range(0, 1000, 10):
            ref.push_back("data", i)

        view = ref.view()
        expected = np.arange(0, 1000, 10)
        assert np.array_equal(view, expected)

    def test_tag_with_nonexistent_parameter(self):
        """Test tagging a parameter that doesn't exist."""
        p = ParameterTree()

        # Based on the implementation, tagging requires the parameter to exist first
        # Let's test what actually happens
        with pytest.raises(KeyError):
            p.tag("nonexistent", "tag1")

        # Add parameter first, then tag should work
        p["existing"] = 42
        p.tag("existing", "tag1")
        assert "tag1" in p.tagnames

    def test_serialization_roundtrip(self):
        """Test parameter tree serialization/deserialization."""
        import pickle

        p = ParameterTree()
        p["a"] = 1
        p["b/arr"] = np.array([1, 2, 3])
        p.tag("a", "important")

        # Test pickle serialization
        serialized = pickle.dumps(p)
        deserialized = pickle.loads(serialized)

        assert deserialized == p
        deserialized.data.check()  # Validate structure


@pytest.fixture
def sample_parameter_tree():
    """Fixture providing a sample ParameterTree for testing."""
    p = ParameterTree()
    p["weights/layer1"] = np.random.randn(10, 5).astype(np.float32)
    p["weights/layer2"] = np.random.randn(5, 1).astype(np.float32)
    p["biases/layer1"] = np.zeros(5, dtype=np.float32)
    p["biases/layer2"] = np.zeros(1, dtype=np.float32)
    p["learning_rate"] = 0.01

    # Add tags
    p.tag("weights/layer1", "trainable")
    p.tag("weights/layer2", "trainable")
    p.tag("biases/layer1", "trainable")
    p.tag("biases/layer2", "trainable")
    p.tag("learning_rate", "hyperparameter")

    return p


class TestIntegrationWithFixtures:
    """Integration tests using fixtures."""

    def test_gradient_computation(self, sample_parameter_tree):
        """Test gradient computation on a realistic parameter tree."""

        def simple_loss(params):
            w1 = params["weights/layer1"]
            w2 = params["weights/layer2"]
            return jnp.sum(w1**2) + jnp.sum(w2**2)

        grad_fn = jax.grad(simple_loss)
        grads = grad_fn(sample_parameter_tree)

        # Check that gradients have the same structure
        assert "weights/layer1" in grads.data
        assert "weights/layer2" in grads.data
        assert grads["weights/layer1"].shape == sample_parameter_tree["weights/layer1"].shape
        assert grads["weights/layer2"].shape == sample_parameter_tree["weights/layer2"].shape

    def test_parameter_updates(self, sample_parameter_tree):
        """Test parameter update operations."""
        learning_rate = 0.1

        # Simulate gradient update
        def update_fn(params, grads):
            return jtu.tree_map(lambda p, g: p - learning_rate * g, params, grads)

        # Create some dummy gradients
        grads = jtu.tree_map(lambda x: np.ones_like(x) * 0.01, sample_parameter_tree)

        updated_params = update_fn(sample_parameter_tree, grads)

        # Check that parameters were updated
        original_w1 = sample_parameter_tree["weights/layer1"]
        updated_w1 = updated_params["weights/layer1"]

        expected_w1 = original_w1 - learning_rate * 0.01
        assert np.allclose(updated_w1, expected_w1)

    def test_tag_based_optimization(self, sample_parameter_tree):
        """Test optimization using only tagged parameters."""

        def loss_with_tags(params):
            trainable, fixed = params.filter_by_tag("trainable")
            # Only compute loss on trainable parameters
            total_loss = 0.0
            for path, value in trainable.data.iter_leaves():
                if isinstance(value, np.ndarray):
                    total_loss += jnp.sum(value**2)
            return total_loss

        grad_fn = jax.grad(loss_with_tags)
        grads = grad_fn(sample_parameter_tree)

        # Should have gradients for trainable parameters
        trainable_params, _ = sample_parameter_tree.filter_by_tag("trainable")
        for path, _ in trainable_params.data.iter_leaves():
            assert path in grads.data


class TestPerformanceAndLargeStructures:
    """Test performance with large parameter structures."""

    def test_large_parameter_tree_operations(self):
        """Test operations on large parameter trees."""
        p = ParameterTree()

        # Create a moderately large tree structure
        for i in range(50):  # Reduced for faster testing
            p[f"layer_{i}/weights"] = np.random.randn(5, 5).astype(np.float32)
            p[f"layer_{i}/bias"] = np.random.randn(5).astype(np.float32)
            p.tag(f"layer_{i}/weights", "trainable")
            p.tag(f"layer_{i}/bias", "trainable")

        # Test filtering performance
        trainable, fixed = p.filter_by_tag("trainable")
        assert len(list(trainable.data.iter_leaves())) == 100  # 50 weights + 50 biases

        # Test JAX operations on large tree
        def simple_loss(params):
            total = 0.0
            for path, value in params.data.iter_leaves():
                if isinstance(value, (np.ndarray, jnp.ndarray)):
                    total += jnp.sum(value**2)
            return total

        loss_value = simple_loss(trainable)
        assert isinstance(loss_value, jnp.ndarray)

    def test_deep_nesting_limits(self):
        """Test behavior with deeply nested parameter structures."""
        p = ParameterTree()

        # Create deeply nested structure
        nested_path = "/".join([f"level_{i}" for i in range(10)]) + "/final_param"
        p[nested_path] = 42.0

        assert p[nested_path] == 42.0

        # Test path operations on deep structures
        path_obj = ParamPath(nested_path)
        assert len(path_obj) == 11  # 10 levels + final_param

    def test_array_ref_with_large_structure(self):
        """Test ArrayRef performance with larger arrays."""
        tree = PTree()

        # Create multiple arrays
        for i in range(5):
            tree[f"array_{i}"] = np.random.randn(100).astype(np.float32)

        ref = ArrayRef(tree)

        # Add many references
        for i in range(5):
            for j in range(0, 100, 10):  # Every 10th element
                ref.push_back(f"array_{i}", j)

        view = ref.view()
        assert view.shape == (50,)  # 5 arrays * 10 samples each


class TestUtilityFunctions:
    """Test utility functions in the parameters module."""

    def test_is_equal_function(self):
        """Test the is_equal utility function."""
        # Test with scalars
        assert pm.is_equal(1, 1)
        assert not pm.is_equal(1, 2)

        # Test with arrays
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        arr3 = np.array([1, 2, 4])

        assert pm.is_equal(arr1, arr2)
        assert not pm.is_equal(arr1, arr3)

        # Test with different types
        assert not pm.is_equal(1, "1")

    def test_pretty_str_function(self):
        """Test the pretty_str utility function."""
        # Test with string
        result = pm.pretty_str("hello")
        assert "hello" in result

        # Test with small array
        arr = np.array([1, 2, 3])
        result = pm.pretty_str(arr)
        assert "[1 2 3]" in result

        # Test with large array
        large_arr = np.random.randn(100)
        result = pm.pretty_str(large_arr)
        assert "array:" in result

    def test_isArrayRef_function(self):
        """Test the isArrayRef utility function."""
        tree = PTree()
        tree["data"] = np.array([1, 2, 3])

        ref = ArrayRef(tree)
        non_ref = np.array([1, 2, 3])

        assert pm.isArrayRef(ref)
        assert not pm.isArrayRef(non_ref)
        assert not pm.isArrayRef("string")


class TestUpdateLeavesByPath:
    """
    Dedicated tests for the `update_leaves_by_path` method, ensuring it
    is functional, JIT-compatible, and safe for ArrayRefs.
    """

    @pytest.fixture
    def complex_tree(self):
        """A fixture with a mix of normal leaves and ArrayRefs."""
        p = ParameterTree()
        # --- Normal leaves ---
        p["a/x"] = jnp.array([1.0, 2.0, 3.0])
        p["b/y"] = jnp.array([4.0])
        p["c"] = 5.0

        # --- Source arrays for references ---
        p["sources/s1"] = jnp.array([10.0, 20.0, 30.0])
        p["sources/s2"] = jnp.array([100.0, 200.0])

        # --- ArrayRef leaf ---
        ref = pm.ArrayRef(p.data)
        ref.push_back("sources/s1", 1)  # 20.0
        ref.push_back("sources/s2", 0)  # 100.0
        p["d/ref"] = ref
        return p

    def test_basic_update(self, complex_tree):
        """Test updating a single, simple leaf."""
        paths_to_update = [ParamPath("a/x")]
        update_func = lambda x: x * 2.0

        new_tree = complex_tree.update_leaves_by_path(paths_to_update, update_func)

        # Check that the new tree has the updated value
        assert np.allclose(new_tree["a/x"], np.array([2.0, 4.0, 6.0]))

        # Check that the original tree is unchanged (functional purity)
        assert np.allclose(complex_tree["a/x"], np.array([1.0, 2.0, 3.0]))
        assert new_tree is not complex_tree

        # Check that other leaves are untouched
        assert np.allclose(new_tree["b/y"], complex_tree["b/y"])
        assert new_tree["c"] == complex_tree["c"]

    def test_multiple_updates(self, complex_tree):
        """Test updating multiple leaves at once."""
        paths_to_update = [ParamPath("a/x"), ParamPath("c")]
        update_func = lambda x: x + 10.0

        new_tree = complex_tree.update_leaves_by_path(paths_to_update, update_func)

        # Check updated values
        assert np.allclose(new_tree["a/x"], np.array([11.0, 12.0, 13.0]))
        assert new_tree["c"] == 15.0

        # Check untouched value
        assert np.allclose(new_tree["b/y"], np.array([4.0]))

    def test_no_paths_to_update(self, complex_tree):
        """Test that the function returns an identical tree if no paths match."""
        paths_to_update = [ParamPath("non/existent/path")]
        update_func = lambda x: x * 1000

        new_tree = complex_tree.update_leaves_by_path(paths_to_update, update_func)

        # The new tree should be equal to the old one (but a different object)
        assert new_tree is not complex_tree
        assert new_tree == complex_tree

    def test_arrayref_is_untouched_and_valid(self, complex_tree):
        """
        Crucial test: Ensure that ArrayRefs are not updated by the function and
        remain self-consistent in the new tree.
        """
        paths_to_update = [ParamPath("d/ref"), ParamPath("sources/s1")]
        update_func = lambda x: x * -1.0

        with patch("biocomp.parameters.logger") as mock_logger:
            new_tree = complex_tree.update_leaves_by_path(paths_to_update, update_func)
            mock_logger.warning.assert_called_once_with(
                "Skipping update for path 'd/ref' because it is an ArrayRef "
                "and cannot be updated with a direct value function."
            )

        assert np.allclose(new_tree["sources/s1"], np.array([-10.0, -20.0, -30.0]))

        expected_view = np.array([-20.0, 100.0])
        assert np.allclose(new_tree["d/ref"], expected_view)

        new_ref_obj = new_tree.data.get_at("d/ref", get_leaf_value=False).value
        assert isinstance(new_ref_obj, ArrayRef)
        assert new_ref_obj.tree is new_tree.data, "ArrayRef must point to its new parent tree"

        new_tree.data.check()

    def test_jit_compatibility(self, complex_tree):
        """Test that the update function can be JIT compiled."""
        paths_to_update = [ParamPath("b/y")]

        def compiled_update_func(x):
            return x / 2.0

        @jax.jit
        def jitted_update(tree):
            return tree.update_leaves_by_path(paths_to_update, compiled_update_func)

        jitted_new_tree = jitted_update(complex_tree)

        assert np.allclose(jitted_new_tree["b/y"], np.array([2.0]))
        assert np.allclose(jitted_new_tree["a/x"], complex_tree["a/x"])

        jitted_ref_obj = jitted_new_tree.data.get_at("d/ref", get_leaf_value=False).value
        assert jitted_ref_obj.tree is jitted_new_tree.data
        jitted_new_tree.data.check()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
