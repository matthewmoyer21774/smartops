"""
Strategy Classifier

Main classifier that runs all strategy detectors on a trader
and returns a comprehensive strategy profile.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trader import Trader
from strategies.base import BaseStrategyDetector, StrategySignals
from strategies.arbitrage import ArbitrageDetector, ArbitrageSignals
from strategies.insider import InsiderDetector, InsiderSignals
from strategies.lazy_positions import LazyPositionDetector, LazyPositionSignals
from strategies.momentum import MomentumDetector, MomentumSignals


@dataclass
class StrategyProfile:
    """Complete strategy profile for a trader."""

    # Identity
    trader_address: str
    trader_name: Optional[str] = None

    # Classification
    primary_strategy: str = "unknown"
    primary_confidence: float = 0.0
    secondary_strategy: Optional[str] = None
    secondary_confidence: float = 0.0

    # All signals
    arbitrage_signals: Optional[ArbitrageSignals] = None
    insider_signals: Optional[InsiderSignals] = None
    lazy_signals: Optional[LazyPositionSignals] = None
    momentum_signals: Optional[MomentumSignals] = None

    # Assessment
    risk_level: str = "medium"  # low, medium, high
    copyable: bool = True
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'trader_address': self.trader_address,
            'trader_name': self.trader_name,
            'primary_strategy': self.primary_strategy,
            'primary_confidence': self.primary_confidence,
            'secondary_strategy': self.secondary_strategy,
            'secondary_confidence': self.secondary_confidence,
            'signals': {
                'arbitrage': self.arbitrage_signals.to_dict() if self.arbitrage_signals else None,
                'insider': self.insider_signals.to_dict() if self.insider_signals else None,
                'lazy_positions': self.lazy_signals.to_dict() if self.lazy_signals else None,
                'momentum': self.momentum_signals.to_dict() if self.momentum_signals else None
            },
            'risk_level': self.risk_level,
            'copyable': self.copyable,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }

    def summary(self) -> str:
        """Human-readable summary."""
        name = self.trader_name or self.trader_address[:10] + "..."
        lines = [
            f"Strategy Profile: {name}",
            f"  Primary: {self.primary_strategy.upper()} ({self.primary_confidence:.0%} confidence)",
        ]
        if self.secondary_strategy:
            lines.append(f"  Secondary: {self.secondary_strategy} ({self.secondary_confidence:.0%})")
        lines.extend([
            f"  Risk Level: {self.risk_level.upper()}",
            f"  Copyable: {'Yes' if self.copyable else 'No'}",
        ])
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings[:3]:
                lines.append(f"    - {w}")
        return "\n".join(lines)


class StrategyClassifier:
    """
    Main classifier that runs all strategy detectors on a trader
    and returns a comprehensive strategy profile.
    """

    STRATEGY_TYPES = {
        'arbitrage': {
            'name': 'Arbitrage Bot',
            'description': 'Automated trading exploiting price inefficiencies',
            'risk': 'low',
            'copyable': False
        },
        'insider': {
            'name': 'Informed Trader',
            'description': 'Apparent information advantage on events',
            'risk': 'high',
            'copyable': False
        },
        'lazy_positions': {
            'name': 'Lazy Position Buyer',
            'description': 'Buys nearly-resolved markets for small profits',
            'risk': 'low',
            'copyable': True
        },
        'momentum': {
            'name': 'Momentum Trader',
            'description': 'Follows trends and established direction',
            'risk': 'medium',
            'copyable': True
        }
    }

    def __init__(self):
        """Initialize all detectors."""
        self.detectors = {
            'arbitrage': ArbitrageDetector(),
            'insider': InsiderDetector(),
            'lazy_positions': LazyPositionDetector(),
            'momentum': MomentumDetector()
        }

    def classify(self, trader: Trader) -> StrategyProfile:
        """
        Analyze trader and return comprehensive strategy profile.

        Args:
            trader: Trader with full transaction history

        Returns:
            StrategyProfile with classification and signals
        """
        profile = StrategyProfile(
            trader_address=trader.address,
            trader_name=trader.name or trader.pseudonym
        )

        if not trader.positions:
            profile.warnings.append("No positions to analyze")
            return profile

        # Run all detectors
        results = {}
        for strategy_type, detector in self.detectors.items():
            signals = detector.analyze(trader)
            results[strategy_type] = signals

        # Store signals
        profile.arbitrage_signals = results.get('arbitrage')
        profile.insider_signals = results.get('insider')
        profile.lazy_signals = results.get('lazy_positions')
        profile.momentum_signals = results.get('momentum')

        # Determine primary and secondary strategies
        ranked = sorted(
            results.items(),
            key=lambda x: x[1].confidence_score,
            reverse=True
        )

        if ranked:
            primary_type, primary_signals = ranked[0]
            profile.primary_strategy = primary_type
            profile.primary_confidence = primary_signals.confidence_score

            if len(ranked) > 1 and ranked[1][1].confidence_score > 0.3:
                profile.secondary_strategy = ranked[1][0]
                profile.secondary_confidence = ranked[1][1].confidence_score

        # Determine risk level and copyability
        strategy_info = self.STRATEGY_TYPES.get(profile.primary_strategy, {})
        profile.risk_level = strategy_info.get('risk', 'medium')
        profile.copyable = strategy_info.get('copyable', True)

        # Aggregate warnings
        all_warnings = []
        for signals in results.values():
            all_warnings.extend(signals.warnings)
        profile.warnings = list(set(all_warnings))[:5]  # Dedupe and limit

        # Add recommendations
        profile.recommendations = self._generate_recommendations(profile, trader)

        return profile

    def _generate_recommendations(self, profile: StrategyProfile, trader: Trader) -> List[str]:
        """Generate actionable recommendations based on profile."""
        recs = []

        strategy = profile.primary_strategy
        confidence = profile.primary_confidence

        if strategy == 'arbitrage':
            if confidence > 0.7:
                recs.append("Do not attempt to copy - requires automated infrastructure")
                recs.append("Study their market selection for insights")
            else:
                recs.append("May have some automated components")

        elif strategy == 'insider':
            if confidence > 0.5:
                recs.append("Track their positions for signals, but verify independently")
                recs.append("Focus on their market category preferences")
            recs.append("High risk - their edge may not be replicable")

        elif strategy == 'lazy_positions':
            if confidence > 0.5:
                recs.append("Can replicate by monitoring nearly-resolved markets")
                recs.append("Low risk strategy but requires patience")
                recs.append("Look for markets at 0.95+ or 0.05- prices")

        elif strategy == 'momentum':
            if confidence > 0.5:
                recs.append("Good candidate for copy trading")
                recs.append("Follow their market category focus")
            if trader.stats.win_rate > 60:
                recs.append(f"Strong track record ({trader.stats.win_rate:.1f}% win rate)")

        # General recommendations based on stats
        if trader.stats.roi_pct > 20:
            recs.append(f"High ROI trader ({trader.stats.roi_pct:.1f}%)")
        if trader.stats.total_trades > 100:
            recs.append("Experienced trader with significant history")

        return recs[:4]  # Limit to 4 recommendations

    def get_strategy_types(self) -> List[dict]:
        """Return list of all strategy types with metadata."""
        return [
            {
                'type': stype,
                'name': info['name'],
                'description': info['description'],
                'risk_level': info['risk'],
                'copyable': info['copyable']
            }
            for stype, info in self.STRATEGY_TYPES.items()
        ]

    def analyze_all_traders(self, traders: List[Trader]) -> List[StrategyProfile]:
        """
        Classify all traders.

        Args:
            traders: List of Trader objects

        Returns:
            List of StrategyProfile objects
        """
        return [self.classify(trader) for trader in traders]

    def find_by_strategy(self, traders: List[Trader], strategy: str,
                        min_confidence: float = 0.3) -> List[StrategyProfile]:
        """
        Find traders using a specific strategy.

        Args:
            traders: List of Trader objects
            strategy: Strategy type to filter by
            min_confidence: Minimum confidence threshold

        Returns:
            List of StrategyProfile objects matching criteria
        """
        profiles = self.analyze_all_traders(traders)
        return [
            p for p in profiles
            if p.primary_strategy == strategy and p.primary_confidence >= min_confidence
        ]

    def get_leaderboard(self, traders: List[Trader], strategy: str,
                       limit: int = 10) -> List[dict]:
        """
        Get top traders for a specific strategy.

        Args:
            traders: List of Trader objects
            strategy: Strategy type
            limit: Max results

        Returns:
            List of trader summaries sorted by confidence
        """
        profiles = self.find_by_strategy(traders, strategy, min_confidence=0.1)
        profiles.sort(key=lambda p: p.primary_confidence, reverse=True)

        results = []
        for profile in profiles[:limit]:
            trader = next((t for t in traders if t.address == profile.trader_address), None)
            if trader:
                results.append({
                    'address': profile.trader_address,
                    'name': profile.trader_name,
                    'confidence': profile.primary_confidence,
                    'pnl': trader.stats.total_pnl,
                    'roi_pct': trader.stats.roi_pct,
                    'win_rate': trader.stats.win_rate,
                    'trades': trader.stats.total_trades,
                    'copyable': profile.copyable,
                    'warnings': profile.warnings[:2]
                })

        return results
