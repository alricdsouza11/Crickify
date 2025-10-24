import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUpIcon, TrendingDownIcon } from "lucide-react";

interface BowlerAnalysis {
  name: string;
  live_stats: {
    overs: number;
    runs_conceded: number;
    wickets: number;
    economy: number;
  };
  historical_stats: {
    economy: number;
    wickets: number;
  };
  insight: string;
}

interface BowlingCardProps {
  bowlersAnalysis: BowlerAnalysis[];
  bowlingTeam?: string;
}

export default function BowlingCard({ bowlersAnalysis, bowlingTeam }: BowlingCardProps) {
  const tableVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.05,
      },
    },
  };

  const rowVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { type: "spring", stiffness: 300, damping: 20 },
    },
  };

  // Format overs to show correct decimal (.1 for 1 ball, .2 for 2 balls, etc.)
  const formattedOvers = (overs: number) => {
    const fullOvers = Math.floor(overs);
    const balls = Math.round((overs - fullOvers) * 6);
    return `${fullOvers}.${balls}`;
  };

  // Compare live economy with historical economy to show trend
  const getEconomyTrend = (bowler: BowlerAnalysis) => {
    // Skip if historical data isn't available or no overs bowled
    if (!bowler.historical_stats.economy || bowler.live_stats.overs === 0) {
      return null;
    }
    
    const diff = bowler.live_stats.economy - bowler.historical_stats.economy;
    
    if (diff >= 1) {
      // Economy is significantly worse than historical
      return {
        icon: <TrendingUpIcon className="h-4 w-4 text-red-500" />,
        color: "text-red-500 dark:text-red-400",
        tooltip: `${diff.toFixed(1)} higher than average`
      };
    } else if (diff <= -1) {
      // Economy is significantly better than historical
      return {
        icon: <TrendingDownIcon className="h-4 w-4 text-green-500" />,
        color: "text-green-500 dark:text-green-400",
        tooltip: `${Math.abs(diff).toFixed(1)} lower than average`
      };
    } else {
      // Economy is close to historical
      return {
        icon: null,
        color: "text-gray-500 dark:text-gray-400",
        tooltip: "Close to average"
      };
    }
  };

  // Only show bowlers who have actually bowled
  const activeBowlers = bowlersAnalysis.filter(bowler => bowler.live_stats.overs > 0);
  
  // Sort bowlers by wickets taken (descending), then by economy (ascending)
  const sortedBowlers = [...activeBowlers].sort((a, b) => {
    if (b.live_stats.wickets !== a.live_stats.wickets) {
      return b.live_stats.wickets - a.live_stats.wickets;
    }
    return a.live_stats.economy - b.live_stats.economy;
  });

  return (
    <Card className="border-none shadow-sm overflow-hidden bg-white dark:bg-gray-900">
      <CardHeader className="bg-blue-50 dark:bg-blue-900/20 border-b border-blue-100 dark:border-blue-800 py-3">
        <CardTitle className="text-blue-700 dark:text-blue-300 text-base font-medium flex justify-between items-center">
          <span>{bowlingTeam || "Bowling"}</span>
          {sortedBowlers.length > 0 && (
            <Badge variant="outline" className="text-xs bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-300 border-blue-200 dark:border-blue-700">
              {sortedBowlers.reduce((total, bowler) => total + bowler.live_stats.wickets, 0)} wickets
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <motion.table
            className="min-w-full divide-y divide-gray-200 dark:divide-gray-700"
            initial="hidden"
            animate="visible"
            variants={tableVariants}
          >
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                >
                  Bowler
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                >
                  O
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                >
                  M
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                >
                  R
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                >
                  W
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                >
                  Econ
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
              {sortedBowlers.length > 0 ? (
                sortedBowlers.map((bowler, index) => {
                  const isCaptain = bowler.name === "JO Holder";
                  const economyTrend = getEconomyTrend(bowler);
                  const isGoodPerformance = bowler.live_stats.wickets >= 2 || 
                    (bowler.live_stats.wickets > 0 && bowler.live_stats.economy < 7);
                  
                  return (
                    <motion.tr
                      key={index}
                      className={`hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200 ${
                        isGoodPerformance ? "bg-green-50/30 dark:bg-green-900/10" : ""
                      }`}
                      variants={rowVariants}
                      whileHover={{ x: 3, transition: { duration: 0.2 } }}
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div>
                            <div className="text-sm font-medium text-gray-900 dark:text-white flex items-center">
                              {bowler.name}
                              {isCaptain && (
                                <Badge 
                                  variant="outline" 
                                  className="ml-1 text-xs py-0 h-5 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800"
                                >
                                  C
                                </Badge>
                              )}
                            </div>
                            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                              {bowler.insight !== "No bowling data available." ? (
                                <span className={economyTrend?.color}>{bowler.insight}</span>
                              ) : null}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-700 dark:text-gray-300">
                        {/* {formattedOvers(bowler.live_stats.overs)} */}
                        {Math.floor(bowler.live_stats.overs).toFixed(1)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-700 dark:text-gray-300">
                        0
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-700 dark:text-gray-300">
                        {bowler.live_stats.runs_conceded}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center text-sm font-medium">
                        <Badge 
                          variant={bowler.live_stats.wickets > 0 ? "default" : "outline"} 
                          className={
                            bowler.live_stats.wickets >= 3 
                              ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 dark:border-green-800" 
                              : bowler.live_stats.wickets > 0 
                                ? "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 dark:border-blue-800"
                                : "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400"
                          }
                        >
                          {bowler.live_stats.wickets}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center text-sm">
                        <div className="flex items-center justify-center">
                          <span className={
                            bowler.live_stats.economy > 10 
                              ? "text-red-600 dark:text-red-400" 
                              : bowler.live_stats.economy < 8 
                                ? "text-green-600 dark:text-green-400" 
                                : "text-orange-600 dark:text-orange-400"
                          }>
                            {bowler.live_stats.economy.toFixed(2)}
                          </span>
                          {economyTrend?.icon && (
                            <span className="ml-1" title={economyTrend.tooltip}>
                              {economyTrend.icon}
                            </span>
                          )}
                        </div>
                      </td>
                    </motion.tr>
                  );
                })
              ) : (
                <motion.tr variants={rowVariants}>
                  <td colSpan={6} className="px-6 py-4 text-center text-sm text-gray-500 dark:text-gray-400">
                    No bowlers have bowled yet
                  </td>
                </motion.tr>
              )}
            </tbody>
          </motion.table>
        </div>
      </CardContent>
    </Card>
  );
}