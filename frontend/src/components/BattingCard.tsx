import { motion, AnimatePresence } from "framer-motion";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ChevronRightIcon, AlertCircleIcon } from "lucide-react";

interface BatterAnalysis {
  name: string;
  live_stats: {
    runs: number;
    balls_faced: number;
    boundaries: number;
    strike_rate: number;
  };
  historical_stats: {
    runs: number;
    strike_rate: number;
  };
  insight: string;
  recent_performance?: number;
}

interface BattingCardProps {
  battersAnalysis: BatterAnalysis[];
  partnerships?: {
    batsmen: string[];
    runs: number;
    balls: number;
    run_rate: number;
  }[];
  matchSummary?: {
    total_runs: number;
    total_wickets: number;
    overs: number;
    current_run_rate: number;
  };
  battingTeam?: string;
}

export default function BattingCard({
  battersAnalysis,
  partnerships,
  matchSummary,
  battingTeam,
}: BattingCardProps) {
  // Determine if a performance is a milestone (50+ runs)
  const isMilestone = (runs: number) => runs >= 50;

  // Get performance indicator
  const getPerformanceIndicator = (batter: BatterAnalysis) => {
    if (!batter.recent_performance) return null;
    
    if (batter.recent_performance >= 1.5) {
      return { label: "In Form", color: "text-green-500 dark:text-green-400" };
    } else if (batter.recent_performance >= 1.0) {
      return { label: "Steady", color: "text-blue-500 dark:text-blue-400" };
    } else if (batter.recent_performance >= 0.5) {
      return { label: "Average", color: "text-orange-500 dark:text-orange-400" };
    } else {
      return { label: "Struggling", color: "text-red-500 dark:text-red-400" };
    }
  };

  // Variants for framer-motion animations
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
    hidden: { opacity: 0, x: -10 },
    visible: {
      opacity: 1,
      x: 0,
      transition: { type: "spring", stiffness: 300, damping: 20 },
    },
  };

  // Separate boundaries into 4s and 6s (estimation based on total boundaries)
  const estimateBoundaries = (total: number, runs: number) => {
    if (total === 0) return { fours: 0, sixes: 0 };
    
    // This is a simple estimation - in a real app you'd have actual data
    const avgRunsPerBoundary = 4.5; // Average between 4 and 6
    const expectedBoundaryRuns = total * avgRunsPerBoundary;
    
    // Adjust the mix to match actual runs better
    const sixRatio = (runs - total * 4) / (total * 2);
    const sixes = Math.round(total * Math.min(Math.max(sixRatio, 0), 1));
    const fours = total - sixes;
    
    return { fours, sixes };
  };

  // Calculate extras
  const calculateExtras = () => {
    if (!matchSummary) return 0;
    
    const battingRuns = battersAnalysis.reduce(
      (sum, batter) => sum + batter.live_stats.runs, 
      0
    );
    
    return matchSummary.total_runs - battingRuns;
  };

  // Distribute extras to w, nb, b and lb
  const distributeExtras = () => {
    const extras = calculateExtras();
    
    // Simple distribution logic - in a real app you'd have actual data
    const w = Math.floor(extras * 0.3);
    const nb = Math.floor(extras * 0.2);
    const b = Math.floor(extras * 0.2);
    const lb = extras - w - nb - b;
    
    return { w, nb, b, lb };
  };

  // Find out if a batter is currently batting (not out)
  const isCurrentlyBatting = (name: string) => {
    if (!partnerships || partnerships.length === 0) return false;
    const currentPartnership = partnerships[partnerships.length - 1];
    return currentPartnership.batsmen.includes(name);
  };

  return (
    <Card className="border-none shadow-lg overflow-hidden bg-white dark:bg-gray-900">
      <CardHeader className="bg-gradient-to-r from-blue-600 to-blue-500 text-white p-4">
        <CardTitle className="text-sm flex items-center justify-between">
          <span className="flex items-center">
            <ChevronRightIcon className="h-4 w-4 mr-2" />
            {battingTeam || "Batting"} Scorecard
          </span>
          {matchSummary && (
            <Badge
              variant="outline"
              className="text-xs text-blue-100 border-blue-300 bg-blue-600/30"
            >
              {matchSummary.total_runs}/{matchSummary.total_wickets} ({matchSummary.overs.toFixed(1)}) | CRR: {matchSummary.current_run_rate.toFixed(2)}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <motion.table
            className="w-full border-collapse"
            initial="hidden"
            animate="visible"
            variants={tableVariants}
          >
            <TableHeader className="bg-blue-50 dark:bg-blue-900/20">
              <TableRow>
                <TableHead className="font-medium text-blue-700 dark:text-blue-300 w-2/5">
                  BATTER
                </TableHead>
                <TableHead className="text-right font-medium text-blue-700 dark:text-blue-300">
                  R
                </TableHead>
                <TableHead className="text-right font-medium text-blue-700 dark:text-blue-300">
                  B
                </TableHead>
                <TableHead className="text-right font-medium text-blue-700 dark:text-blue-300">
                  4s
                </TableHead>
                <TableHead className="text-right font-medium text-blue-700 dark:text-blue-300">
                  6s
                </TableHead>
                <TableHead className="text-right font-medium text-blue-700 dark:text-blue-300">
                  S/R
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <AnimatePresence>
                {battersAnalysis.map((batter, index) => {
                  if (batter.live_stats.balls_faced === 0) return null;
                  
                  const { fours, sixes } = estimateBoundaries(
                    batter.live_stats.boundaries,
                    batter.live_stats.runs
                  );
                  const milestone = isMilestone(batter.live_stats.runs);
                  const performance = getPerformanceIndicator(batter);
                  const batting = isCurrentlyBatting(batter.name);
                  
                  return (
                    <motion.tr
                      key={index}
                      variants={rowVariants}
                      initial="hidden"
                      animate="visible"
                      transition={{ delay: index * 0.05 }}
                      whileHover={{
                        backgroundColor: "rgba(59, 130, 246, 0.05)",
                        transition: { duration: 0.2 },
                      }}
                      className={`${
                        milestone ? "bg-blue-50/40 dark:bg-blue-900/10" : ""
                      } ${
                        batting ? "border-l-4 border-green-400 dark:border-green-600" : ""
                      }`}
                    >
                      <TableCell className="font-medium py-4">
                        <div className="flex flex-col">
                          <div className="font-medium text-gray-900 dark:text-white flex items-center">
                            {batter.name}
                            {batter.name === "RG Sharma" && (
                              <Badge
                                variant="outline"
                                className="ml-2 text-xs py-0 h-5 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800"
                              >
                                C
                              </Badge>
                            )}
                            {milestone && (
                              <span className="ml-2 text-yellow-500 dark:text-yellow-400">
                                ★
                              </span>
                            )}
                          </div>
                          <motion.div
                            className="flex items-center text-xs mt-1"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.3 }}
                          >
                            {batting ? (
                              <span className="text-green-500 dark:text-green-400 font-medium">
                                Not Out
                              </span>
                            ) : (
                              <span className="text-gray-500 dark:text-gray-400">
                                Out
                              </span>
                            )}
                            {performance && (
                              <span className={`ml-2 ${performance.color}`}>
                                • {performance.label}
                              </span>
                            )}
                          </motion.div>
                        </div>
                      </TableCell>
                      <TableCell className="text-right font-semibold text-gray-900 dark:text-white">
                        {milestone ? (
                          <motion.span
                            initial={{ scale: 1 }}
                            whileHover={{ scale: 1.1 }}
                            className="text-blue-600 dark:text-blue-400"
                          >
                            {batter.live_stats.runs}
                          </motion.span>
                        ) : (
                          batter.live_stats.runs
                        )}
                      </TableCell>
                      <TableCell className="text-right text-gray-600 dark:text-gray-400">
                        {batter.live_stats.balls_faced}
                      </TableCell>
                      <TableCell className="text-right text-gray-600 dark:text-gray-400">
                        {fours}
                      </TableCell>
                      <TableCell className="text-right text-gray-600 dark:text-gray-400">
                        {sixes}
                      </TableCell>
                      <TableCell className="text-right text-gray-600 dark:text-gray-400">
                        {batter.live_stats.strike_rate.toFixed(2)}
                      </TableCell>
                    </motion.tr>
                  );
                })}
                <motion.tr
                  initial="hidden"
                  animate="visible"
                  variants={rowVariants}
                  transition={{ delay: battersAnalysis.length * 0.05 }}
                  className="bg-blue-50/50 dark:bg-blue-900/5"
                >
                  <TableCell colSpan={1} className="font-medium py-4">
                    <div className="flex items-center text-blue-600 dark:text-blue-400">
                      <AlertCircleIcon className="h-4 w-4 mr-2" />
                      Extras
                    </div>
                  </TableCell>
                  <TableCell className="text-right font-semibold text-blue-600 dark:text-blue-400">
                    {calculateExtras()}
                  </TableCell>
                  <TableCell colSpan={4} className="text-right">
                    <div className="flex flex-wrap justify-end gap-2 text-sm text-gray-500 dark:text-gray-400">
                      <Badge
                        variant="outline"
                        className="bg-blue-50 dark:bg-blue-900/10 text-gray-600 dark:text-gray-400 border-blue-100 dark:border-blue-900/20"
                      >
                        W {distributeExtras().w}
                      </Badge>
                      <Badge
                        variant="outline"
                        className="bg-blue-50 dark:bg-blue-900/10 text-gray-600 dark:text-gray-400 border-blue-100 dark:border-blue-900/20"
                      >
                        NB {distributeExtras().nb}
                      </Badge>
                      <Badge
                        variant="outline"
                        className="bg-blue-50 dark:bg-blue-900/10 text-gray-600 dark:text-gray-400 border-blue-100 dark:border-blue-900/20"
                      >
                        B {distributeExtras().b}
                      </Badge>
                      <Badge
                        variant="outline"
                        className="bg-blue-50 dark:bg-blue-900/10 text-gray-600 dark:text-gray-400 border-blue-100 dark:border-blue-900/20"
                      >
                        LB {distributeExtras().lb}
                      </Badge>
                    </div>
                  </TableCell>
                </motion.tr>
              </AnimatePresence>
            </TableBody>
          </motion.table>
        </div>
        
        {/* Partnerships section */}
        {partnerships && partnerships.length > 0 && (
          <div className="mt-4 p-4 bg-blue-50/50 dark:bg-blue-900/5">
            <h3 className="text-sm font-medium text-blue-700 dark:text-blue-300 mb-2">
              Partnerships
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {partnerships.map((partnership, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white dark:bg-gray-800 p-2 rounded shadow-sm"
                >
                  <div className="flex justify-between items-center">
                    <span className="text-xs font-medium">
                      {partnership.batsmen.join(" & ")}
                    </span>
                    <Badge variant="secondary" className="text-xs">
                      {partnership.runs} runs ({partnership.balls} balls)
                    </Badge>
                  </div>
                  <div className="mt-1 text-xs text-gray-500">
                    Run rate: {partnership.run_rate.toFixed(2)}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}