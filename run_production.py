#!/usr/bin/env python3
"""
AEGIS-2 Production Deployment Script

Runs the complete AEGIS-2 Apex system with production-level configuration.
This includes all layers: Core Agent, Population Dynamics, Meta-Evolution,
Self-Modification, Genesis, and Singularity.
"""

import sys
import signal
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from apex.system import AEGIS2Apex, ApexConfig


# Production configuration
PRODUCTION_CONFIG = ApexConfig(
    # Population settings - larger for production
    population_size=20,

    # Genesis (source code analysis) - enabled
    genesis_enabled=True,
    genesis_rate=50,

    # Singularity (recursive self-improvement) - enabled with safeguards
    singularity_enabled=True,
    singularity_rate=100,
    max_recursive_depth=3,

    # Safety constraints active
    constitution_active=True
)

# Production run parameters
PRODUCTION_CYCLES = 10000  # Long production run
CHECKPOINT_INTERVAL = 500  # Save state every N cycles
VERBOSE = True


class ProductionRunner:
    """Manages the production deployment of AEGIS-2."""

    def __init__(self):
        self.apex = None
        self.running = True
        self.start_time = None

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n  [SIGNAL] Graceful shutdown initiated...")
        self.running = False

    def run(self):
        """Run the production system."""
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.start_time = time.time()

        print("""
================================================================================
                     AEGIS-2 PRODUCTION DEPLOYMENT
================================================================================

Starting production run with configuration:
  - Population Size: {pop_size}
  - Genesis Enabled: {genesis}
  - Singularity Enabled: {sing}
  - Total Cycles: {cycles}
  - Checkpoint Interval: {checkpoint}

Initializing system...
""".format(
            pop_size=PRODUCTION_CONFIG.population_size,
            genesis=PRODUCTION_CONFIG.genesis_enabled,
            sing=PRODUCTION_CONFIG.singularity_enabled,
            cycles=PRODUCTION_CYCLES,
            checkpoint=CHECKPOINT_INTERVAL
        ))

        # Initialize the Apex system
        self.apex = AEGIS2Apex(
            name="production",
            config=PRODUCTION_CONFIG,
            source_dir=Path(__file__).parent
        )

        print("  System initialized successfully.\n")
        print("=" * 80)
        print()

        # Run the main loop
        cycle = 0
        total_emergence = 0

        try:
            while self.running and cycle < PRODUCTION_CYCLES:
                # Run a step
                result = self.apex.step()
                cycle += 1

                # Track emergence
                emergence_count = result.get('ultimate', {}).get('emergence', 0)
                total_emergence += emergence_count

                # Progress output
                if VERBOSE and cycle % 10 == 0:
                    status = self.apex.status()
                    elapsed = time.time() - self.start_time
                    rate = cycle / elapsed if elapsed > 0 else 0

                    print(f"  Cycle {cycle:>6}/{PRODUCTION_CYCLES} | "
                          f"Pop {status['population']['size']:>2} | "
                          f"Fit {status['population']['mean_fitness']:>.4f} | "
                          f"Prims {status['meta']['primitives']:>4} | "
                          f"Emrg {status['emergence']:>5} | "
                          f"Rate {rate:>.1f}/s")

                # Checkpoint
                if cycle % CHECKPOINT_INTERVAL == 0:
                    print(f"\n  [CHECKPOINT] Saving state at cycle {cycle}...")
                    save_path = self.apex.save()
                    print(f"  [CHECKPOINT] State saved to: {save_path}\n")

        except Exception as e:
            print(f"\n  [ERROR] {e}")
            print("  Attempting to save state before exit...")

        finally:
            # Final save and cleanup
            print("\n" + "=" * 80)
            print("  PRODUCTION RUN COMPLETE")
            print("=" * 80)

            if self.apex:
                # Print final status
                self._print_final_report()

                # Save final state
                print("\n  Saving final state...")
                save_path = self.apex.save()
                print(f"  State saved to: {save_path}")

                # Cleanup
                self.apex.cleanup()

        return self.apex

    def _print_final_report(self):
        """Print comprehensive final report."""
        status = self.apex.status()
        elapsed = time.time() - self.start_time

        print(f"""
  FINAL STATISTICS
  ----------------
  Runtime:           {elapsed:.1f} seconds
  Total Cycles:      {status['cycle']}
  Cycles/Second:     {status['cycle']/elapsed:.2f}

  POPULATION
  ----------
  Final Size:        {status['population']['size']}
  Total Created:     {status['population']['created']}
  Mean Fitness:      {status['population']['mean_fitness']:.6f}

  META-EVOLUTION
  --------------
  Primitives:        {status['meta']['primitives']}
  Fitness Comps:     {status['meta']['fitness_components']}

  SOURCE CODE
  -----------
  Genesis Mods:      {status['genesis']['modifications']}
  Code Fragments:    {status['genesis']['fragments']}
  Singularity Depth: {status['singularity']['depth']}

  EMERGENCE
  ---------
  Total Events:      {status['emergence']}
""")

        print("  Emergence by Type:")
        for etype, count in sorted(status['emergence_by_type'].items(),
                                   key=lambda x: -x[1])[:10]:
            print(f"    {etype:35s}: {count:>5}")


def main():
    """Main entry point for production deployment."""
    runner = ProductionRunner()
    return runner.run()


if __name__ == "__main__":
    main()
