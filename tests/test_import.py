import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestImport(unittest.TestCase):
    def test_vistavu_import(self):
        """Test that the main package and submodules can be imported."""
        try:
            import vistavu
            import vistavu.rules
            import vistavu.rules.extraction
            print("✅ vistavu.rules.extraction imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import vistavu modules: {e}")

    def test_dolphin_import(self):
        """Test that dolphin submodule exists."""
        try:
            import vistavu.dolphin
            print("✅ vistavu.dolphin imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import dolphin: {e}")

if __name__ == '__main__':
    unittest.main()
