import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Pie,
  PieChart,
} from "recharts";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  ChevronUpIcon,
  ChevronDownIcon,
  BarChart3Icon,
  LineChartIcon,
  AlertTriangleIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  ArrowRightIcon,
  CheckCircleIcon,
  ArrowUpIcon,
  ArrowDownIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const scaleIn = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 20,
    },
  },
};

// Helper functions
const getBadgeColorForStrikeRate = (sr: any) => {
  if (sr === 0)
    return "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400";
  if (sr > 180)
    return "bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400";
  if (sr > 150)
    return "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/20 dark:text-emerald-400";
  if (sr > 120)
    return "bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400";
  if (sr > 100)
    return "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-400";
  return "bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-400";
};

const getBadgeColorForEconomy = (economy: any) => {
  if (economy === 0)
    return "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400";
  if (economy < 6)
    return "bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400";
  if (economy < 8)
    return "bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400";
  if (economy < 10)
    return "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-400";
  return "bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-400";
};

const getFormPercentage = (batter: any) => {
  if (!batter.recent_performance) return 50;
  // Scale from 0-2 to 0-100
  return Math.min(100, Math.max(0, batter.recent_performance * 50));
};

const getFormColor = (batter: any) => {
  if (!batter.recent_performance) return "bg-gray-400";
  if (batter.recent_performance > 1.5) return "bg-green-500";
  if (batter.recent_performance > 1.0) return "bg-blue-500";
  if (batter.recent_performance > 0.5) return "bg-yellow-500";
  return "bg-red-500";
};

const getMomentumDescription = (momentum: any) => {
  const num = parseFloat(momentum);
  if (num > 65) return "Strong momentum";
  if (num > 55) return "Favorable momentum";
  if (num > 45) return "Balanced momentum";
  if (num > 35) return "Challenging position";
  return "Struggling momentum";
};

const parseRecommendations = (rawRecommendations: any) => {
  if (!rawRecommendations) return [];

  // Split the string by newline and filter out empty lines and headers
  const lines = rawRecommendations
    .split("\n")
    .filter(
      (line: any) =>
        line.trim() !== "" &&
        !line.includes("Recommendations:") &&
        !line.startsWith("Topic:") &&
        !line.startsWith("Actions:") &&
        !line.startsWith("Purpose:")
    );

  // Take only action items (lines that start with a number or have "Actions:" prefix)
  const recommendations = [];
  let currentRec = "";

  for (const line of lines) {
    if (/^\d+\./.test(line.trim())) {
      if (currentRec) recommendations.push(currentRec);
      currentRec = line.trim().replace(/^\d+\.\s*Topic:\s*/, "");
    } else if (line.includes("Actions:")) {
      currentRec += ": " + line.trim().replace("Actions:", "").trim();
    }
  }

  if (currentRec) recommendations.push(currentRec);

  // Return max 5 recommendations
  return recommendations;
};

const AnimatedProgressBar = ({ value, colorClass }: any) => (
  <motion.div
    initial={{ width: 0 }}
    animate={{ width: `${value}%` }}
    transition={{ duration: 0.5, ease: "easeOut" }}
    className={`h-2 rounded-full ${colorClass || "bg-blue-500"}`}
  />
);

const StatCard = ({ label, value, description, color, icon }: any) => (
  <div
    className={`rounded-lg p-4 ${
      color || "bg-gradient-to-br from-blue-500 to-blue-600"
    } text-white shadow-md`}
  >
    <div className="flex justify-between">
      <p className="text-xs font-medium text-blue-100">{label}</p>
      {icon}
    </div>
    <p className="text-2xl font-bold mt-2">{value}</p>
    {description && <p className="text-xs text-blue-100 mt-1">{description}</p>}
  </div>
);

const BatterInsightCard = ({ batter, index }: any) => {
  if (!batter.live_stats.balls_faced) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1, duration: 0.3 }}
    >
      <Card className="border border-blue-100 dark:border-blue-900 overflow-hidden">
        <CardHeader className="pb-2">
          <div className="flex justify-between items-center">
            <CardTitle className="text-base font-medium">
              {batter.name}
            </CardTitle>
            <Badge
              className={cn(
                getBadgeColorForStrikeRate(batter.live_stats.strike_rate)
              )}
            >
              SR: {batter.live_stats.strike_rate.toFixed(1)}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-2 mb-4">
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">Runs</p>
              <p className="font-semibold">{batter.live_stats.runs}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">Balls</p>
              <p className="font-semibold">{batter.live_stats.balls_faced}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Boundaries
              </p>
              <p className="font-semibold">
                {batter.live_stats.boundaries || 0}
              </p>
            </div>
          </div>

          {batter.insight && (
            <div className="bg-blue-50 dark:bg-blue-900/10 p-3 rounded-md text-sm">
              {batter.insight}
            </div>
          )}

          {batter.historical_stats && (
            <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-800">
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                Career Stats
              </p>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Runs
                  </p>
                  <p className="font-semibold">
                    {batter.historical_stats.runs}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Strike Rate
                  </p>
                  <p className="font-semibold">
                    {batter.historical_stats.strike_rate}
                  </p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
};

const BowlerInsightCard = ({ bowler, index }: any) => {
  if (!bowler.live_stats.overs) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1, duration: 0.3 }}
    >
      <Card className="border border-blue-100 dark:border-blue-900 overflow-hidden">
        <CardHeader className="pb-2">
          <div className="flex justify-between items-center">
            <CardTitle className="text-base font-medium">
              {bowler.name}
            </CardTitle>
            <Badge
              className={cn(getBadgeColorForEconomy(bowler.live_stats.economy))}
            >
              Eco: {bowler.live_stats.economy.toFixed(2)}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-2 mb-4">
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">Overs</p>
              <p className="font-semibold">
                {bowler.live_stats.overs.toFixed(1)}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">Runs</p>
              <p className="font-semibold">{bowler.live_stats.runs_conceded}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Wickets
              </p>
              <p className="font-semibold">{bowler.live_stats.wickets}</p>
            </div>
          </div>

          {bowler.insight && (
            <div className="bg-blue-50 dark:bg-blue-900/10 p-3 rounded-md text-sm">
              {bowler.insight}
            </div>
          )}

          {bowler.historical_stats && bowler.historical_stats.economy > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-800">
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                Career Stats
              </p>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Economy
                  </p>
                  <p className="font-semibold">
                    {bowler.historical_stats.economy.toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Wickets
                  </p>
                  <p className="font-semibold">
                    {bowler.historical_stats.wickets}
                  </p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
};

const PredictionsTab = ({
  summary,
  batting_momentum,
  bowling_momentum,
}: any) => {
  // Create data for the pie chart using the momentum values
  const momentumData = [
    { name: "Batting", value: batting_momentum, fill: "#3b82f6" },
    { name: "Bowling", value: bowling_momentum, fill: "#ef4444" },
  ];

  return (
    <motion.div>
      <Card className="overflow-hidden border-none shadow-lg">
        <CardHeader className="pb-2 border-b border-gray-100 dark:border-gray-800">
          <CardTitle className="text-lg font-semibold text-blue-700 dark:text-blue-400">
            Momentum Analysis
          </CardTitle>
          <CardDescription>
            Batting & Bowling Momentum Analysis and Outcome Predictions
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          <div className="flex flex-col md:flex-row items-center justify-center gap-8">
            <motion.div
              className="w-full md:w-1/2 flex flex-col items-center"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              <h3 className="text-lg font-medium mb-3 text-center">
                Momentum Distribution
              </h3>
              <div className="w-60 h-60">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={momentumData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={70}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, percent }) =>
                        `${name}: ${(percent * 100).toFixed(1)}%`
                      }
                    >
                      {momentumData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(value: any) => `${value.toFixed(1)}%`}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-between w-full mt-2">
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-blue-600 rounded-full mr-1"></div>
                  <span className="text-xs font-medium">Batting</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-red-600 rounded-full mr-1"></div>
                  <span className="text-xs font-medium">Bowling</span>
                </div>
              </div>
            </motion.div>

            <div className="w-full md:w-1/2 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg mt-6 md:mt-0">
              <h3 className="text-lg font-medium mb-3 text-center">
                Outcome Predictions
              </h3>
              <motion.div
                className="space-y-3"
                variants={staggerContainer}
                initial="hidden"
                animate="visible"
              >
                <motion.div
                  variants={fadeInUp}
                  className="flex justify-between"
                >
                  <span className="text-sm">Batting Momentum</span>
                  <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400">
                    {batting_momentum.toFixed(1)}%
                  </Badge>
                </motion.div>
                <motion.div
                  variants={fadeInUp}
                  className="flex justify-between"
                >
                  <span className="text-sm">Bowling Momentum</span>
                  <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400">
                    {bowling_momentum.toFixed(1)}%
                  </Badge>
                </motion.div>
                <motion.div
                  variants={fadeInUp}
                  className="flex justify-between"
                >
                  <span className="text-sm">Current Run Rate</span>
                  <span className="text-sm font-medium">
                    {summary.current_run_rate.toFixed(2)}
                  </span>
                </motion.div>
                <motion.div
                  variants={fadeInUp}
                  className="flex justify-between"
                >
                  <span className="text-sm">Projected Score</span>
                  <span className="text-sm font-medium">
                    {Math.round(summary.current_run_rate * 20)}
                  </span>
                </motion.div>
              </motion.div>
            </div>
          </div>

          <motion.div
            className="mt-8 p-4 border border-yellow-200 bg-yellow-50 dark:bg-yellow-900/10 dark:border-yellow-800 rounded-lg"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.3 }}
          >
            <div className="flex items-start">
              <AlertTriangleIcon className="h-5 w-5 text-yellow-500 mr-2 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                  Analysis Disclaimer
                </p>
                <p className="text-xs text-yellow-700 dark:text-yellow-300 mt-1">
                  This momentum analysis is based on live match data, historical
                  performance, and statistical models. Metrics may change as the
                  match progresses.
                </p>
              </div>
            </div>
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

const MatchSummary = ({
  summary,
  batting_momentum,
  bowling_momentum,
  recommendations,
}: any) => {
  // Create run rate data from over metrics
  const runRateData =
    summary && summary.run_rate_by_over
      ? Object.keys(summary.run_rate_by_over).map((over) => ({
          name: "Over " + over,
          run_rate: Number(summary.run_rate_by_over[over].toFixed(2)),
        }))
      : [];

  // If no run_rate_by_over data, use current run rate
  if (runRateData.length === 0 && summary) {
    runRateData.push({
      name: "Current",
      run_rate: Number(summary.current_run_rate.toFixed(2)),
    });
  }

  return (
    <motion.div
      className="space-y-6"
      variants={staggerContainer}
      initial="hidden"
      animate="visible"
    >
      <motion.div>
        <Card className="overflow-hidden border-none shadow-lg">
          <CardContent className="p-6">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              <motion.div variants={scaleIn}>
                <StatCard
                  label="Batting Momentum"
                  value={
                    batting_momentum ? `${batting_momentum.toFixed(1)}%` : "N/A"
                  }
                  description={getMomentumDescription(batting_momentum || 0)}
                  color="bg-gradient-to-br from-blue-500 to-blue-600"
                  icon={<TrendingUpIcon className="h-5 w-5 text-blue-100" />}
                />
              </motion.div>
              <motion.div variants={scaleIn}>
                <StatCard
                  label="Bowling Momentum"
                  value={
                    bowling_momentum ? `${bowling_momentum.toFixed(1)}%` : "N/A"
                  }
                  description={getMomentumDescription(bowling_momentum || 0)}
                  color="bg-gradient-to-br from-blue-500 to-blue-600"
                  icon={<TrendingUpIcon className="h-5 w-5 text-blue-100" />}
                />
              </motion.div>
              <motion.div variants={scaleIn}>
                <StatCard
                  label="Total Runs"
                  value={summary?.total_runs || 0}
                  description={"Runs scored"}
                  icon={<BarChart3Icon className="h-5 w-5 text-blue-100" />}
                />
              </motion.div>
              <motion.div variants={scaleIn}>
                <StatCard
                  label="Wickets"
                  value={summary?.total_wickets || 0}
                  description={"Wickets taken"}
                  icon={<ChevronDownIcon className="h-5 w-5 text-blue-100" />}
                />
              </motion.div>
              <motion.div variants={scaleIn}>
                <StatCard
                  label="Run Rate"
                  value={(summary?.current_run_rate || 0).toFixed(2)}
                  description={"RPO (Runs Per Over)"}
                  icon={<LineChartIcon className="h-5 w-5 text-blue-100" />}
                />
              </motion.div>
              <motion.div variants={scaleIn}>
                <StatCard
                  label="Overs"
                  value={Math.floor(summary?.overs) || 0}
                  description={"Total overs bowled"}
                  icon={<ChevronUpIcon className="h-5 w-5 text-blue-100" />}
                />
              </motion.div>
            </div>

            <motion.div className="mt-8">
              <Card className="border border-blue-100 dark:border-blue-900">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base font-medium text-blue-700 dark:text-blue-400">
                    Over-by-Over Run Rate Progression
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={runRateData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                        <XAxis dataKey="name" />
                        <YAxis label={{
                      value: "Run Rate",
                      angle: -90,
                      position: "insideLeft",
                    }} />
                        <Tooltip
                          contentStyle={{
                            color: "#000000",
                            backgroundColor: "rgba(255, 255, 255, 0.95)",
                            borderRadius: "8px",
                            boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
                            border: "1px solid #e0e0e0",
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="run_rate"
                          stroke="#3b82f6"
                          strokeWidth={3}
                          activeDot={{
                            r: 8,
                            fill: "#2563eb",
                            stroke: "#fff",
                            strokeWidth: 2,
                          }}
                          dot={{
                            r: 6,
                            fill: "#3b82f6",
                            stroke: "#fff",
                            strokeWidth: 2,
                          }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* <motion.div className="mt-6">
              <Card className="border border-blue-100 dark:border-blue-900">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base font-medium text-blue-700 dark:text-blue-400 flex items-center">
                    <CheckCircleIcon className="h-4 w-4 mr-2" />
                    Smart Recommendations
                  </CardTitle>
                  <CardDescription>
                    Based on current match analytics
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {recommendations &&
                      recommendations.map((rec: any, i: any) => (
                        <motion.div
                          key={i}
                          className="flex items-start p-3 rounded-lg bg-blue-50 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-800"
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.1, duration: 0.3 }}
                          whileHover={{ x: 3 }}
                        >
                          <ArrowRightIcon className="h-5 w-5 text-blue-500 dark:text-blue-400 mt-0.5 mr-2 flex-shrink-0" />
                          <p className="text-sm text-gray-700 dark:text-gray-300">
                            {rec}
                          </p>
                        </motion.div>
                      ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div> */}
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
};

// BattersAnalysis Component
const BattersAnalysis = ({ batters }: any) => {
  // Filter out batters with no balls faced (likely not yet batted)
  const activeBatters =
    batters?.filter((batter: any) => batter.live_stats?.balls_faced > 0) || [];

  return (
    <motion.div
      className="space-y-6"
      variants={staggerContainer}
      initial="hidden"
      animate="visible"
    >
      <Card className="overflow-hidden border-none shadow-lg">
        <CardHeader className="pb-2 border-b border-gray-100 dark:border-gray-800">
          <CardTitle className="text-lg font-semibold text-blue-700 dark:text-blue-400">
            Batting Analysis
          </CardTitle>
          <CardDescription>
            Performance insights for each batter
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          {/* Active Batters Section */}
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-4 flex items-center">
              <TrendingUpIcon className="h-5 w-5 mr-2 text-blue-500" />
              Current Batters
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {activeBatters.length > 0 ? (
                activeBatters
                  .filter((batter: any) => batter.is_batting)
                  .map((batter: any, index: any) => (
                    <BatterInsightCard
                      key={batter.name}
                      batter={batter}
                      index={index}
                    />
                  ))
              ) : (
                <p className="text-gray-500 col-span-2">
                  No active batters at the moment.
                </p>
              )}
            </div>
          </div>

          {/* All Batters Section */}
          <div>
            <h3 className="text-lg font-medium mb-4">
              All Batters Performance
            </h3>

            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Batter</TableHead>
                  <TableHead className="text-right">Runs</TableHead>
                  <TableHead className="text-right">Balls</TableHead>
                  <TableHead className="text-right">SR</TableHead>
                  <TableHead className="text-right">Boundaries</TableHead>
                  <TableHead className="text-right">Form</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {batters && batters.length > 0 ? (
                  batters.map((batter: any) => (
                    <TableRow key={batter.name}>
                      <TableCell className="font-medium">
                        <div className="flex items-center">
                          {batter.is_batting && (
                            <Badge
                              variant="outline"
                              className="mr-2 bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400"
                            >
                              At Crease
                            </Badge>
                          )}
                          {batter.name}
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        {batter.live_stats?.runs || 0}
                      </TableCell>
                      <TableCell className="text-right">
                        {batter.live_stats?.balls_faced || 0}
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge
                          className={getBadgeColorForStrikeRate(
                            batter.live_stats?.strike_rate || 0
                          )}
                        >
                          {(batter.live_stats?.strike_rate || 0).toFixed(1)}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        {batter.live_stats?.boundaries || 0}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5 overflow-hidden">
                          <AnimatedProgressBar
                            value={getFormPercentage(batter)}
                            colorClass={getFormColor(batter)}
                          />
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell
                      colSpan={6}
                      className="text-center py-4 text-gray-500"
                    >
                      No batting data available.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>

            {/* Batting Insights Accordion */}
            <Accordion type="single" collapsible className="mt-6">
              <AccordionItem value="insights">
                <AccordionTrigger className="text-blue-600 dark:text-blue-400">
                  See Detailed Batting Insights
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-4 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                    {batters && batters.length > 0 ? (
                      batters
                        .filter((batter: any) => batter.insight)
                        .map((batter: any, index: any) => (
                          <div
                            key={batter.name}
                            className="p-3 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-100 dark:border-gray-700"
                          >
                            <p className="font-medium text-blue-700 dark:text-blue-400 mb-1">
                              {batter.name}
                            </p>
                            <p className="text-sm text-gray-700 dark:text-gray-300">
                              {batter.insight}
                            </p>
                          </div>
                        ))
                    ) : (
                      <p className="text-gray-500">No insights available.</p>
                    )}
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

// BowlersAnalysis Component
const BowlersAnalysis = ({ bowlers }: any) => {
  // Filter out bowlers with no overs bowled
  const activeBowlers =
    bowlers?.filter((bowler: any) => bowler.live_stats?.overs > 0) || [];

  return (
    <motion.div
      className="space-y-6"
      variants={staggerContainer}
      initial="hidden"
      animate="visible"
    >
      <Card className="overflow-hidden border-none shadow-lg">
        <CardHeader className="pb-2 border-b border-gray-100 dark:border-gray-800">
          <CardTitle className="text-lg font-semibold text-blue-700 dark:text-blue-400">
            Bowling Analysis
          </CardTitle>
          <CardDescription>
            Performance insights for each bowler
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          {/* Current Bowler Section */}
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-4 flex items-center">
              <TrendingUpIcon className="h-5 w-5 mr-2 text-blue-500" />
              Current Bowler
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {activeBowlers.length > 0 ? (
                activeBowlers
                  .filter((bowler: any) => bowler.is_bowling)
                  .map((bowler: any, index: any) => (
                    <BowlerInsightCard
                      key={bowler.name}
                      bowler={bowler}
                      index={index}
                    />
                  ))
              ) : (
                <p className="text-gray-500 col-span-2">
                  No active bowler at the moment.
                </p>
              )}
            </div>
          </div>

          {/* Economy Rate Chart */}
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-4">
              Economy Rate Comparison
            </h3>

            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={activeBowlers.map((bowler: any) => ({
                    name: bowler.name,
                    economy: bowler.live_stats.economy,
                    wickets: bowler.live_stats.wickets,
                  }))}
                  margin={{ top: 10, right: 10, left: 10, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis
                    dataKey="name"
                    angle={-45}
                    textAnchor="end"
                    height={70}
                  />
                  <YAxis
                    label={{
                      value: "Economy",
                      angle: -90,
                      position: "insideLeft",
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      color: "#000000",
                      backgroundColor: "rgba(255, 255, 255, 0.95)",
                      borderRadius: "8px",
                      boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
                      border: "1px solid #e0e0e0",
                    }}
                    formatter={(value: any, name, props) => {
                      if (name === "economy")
                        return [`${value.toFixed(2)}`, "Economy Rate"];
                      return [value, name];
                    }}
                  />
                  <Bar dataKey="economy" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                    {activeBowlers.map((entry: any, index: any) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          entry.live_stats.economy < 7
                            ? "#22c55e"
                            : entry.live_stats.economy < 9
                            ? "#3b82f6"
                            : "#ef4444"
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* All Bowlers Table */}
          <div>
            <h3 className="text-lg font-medium mb-4">
              All Bowlers Performance
            </h3>

            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Bowler</TableHead>
                  <TableHead className="text-right">Overs</TableHead>
                  <TableHead className="text-right">Runs</TableHead>
                  <TableHead className="text-right">Wickets</TableHead>
                  <TableHead className="text-right">Economy</TableHead>
                  <TableHead className="text-right">Dot Balls</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {bowlers && bowlers.length > 0 ? (
                  bowlers.map((bowler: any) => (
                    <TableRow key={bowler.name}>
                      <TableCell className="font-medium">
                        <div className="flex items-center">
                          {bowler.is_bowling && (
                            <Badge
                              variant="outline"
                              className="mr-2 bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400"
                            >
                              Bowling
                            </Badge>
                          )}
                          {bowler.name}
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        {Math.floor(bowler.live_stats?.overs).toFixed(1) || 0.0}
                      </TableCell>
                      <TableCell className="text-right">
                        {bowler.live_stats?.runs_conceded || 0}
                      </TableCell>
                      <TableCell className="text-right">
                        {bowler.live_stats?.wickets || 0}
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge
                          className={getBadgeColorForEconomy(
                            bowler.live_stats?.economy || 0
                          )}
                        >
                          {(bowler.live_stats?.economy || 0).toFixed(2)}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        {bowler.live_stats?.dot_balls || 0}
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell
                      colSpan={6}
                      className="text-center py-4 text-gray-500"
                    >
                      No bowling data available.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>

            {/* Bowling Insights Accordion */}
            <Accordion type="single" collapsible className="mt-6">
              <AccordionItem value="insights">
                <AccordionTrigger className="text-blue-600 dark:text-blue-400">
                  See Detailed Bowling Insights
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-4 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                    {bowlers && bowlers.length > 0 ? (
                      bowlers
                        .filter((bowler: any) => bowler.insight)
                        .map((bowler: any, index: any) => (
                          <div
                            key={bowler.name}
                            className="p-3 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-100 dark:border-gray-700"
                          >
                            <p className="font-medium text-blue-700 dark:text-blue-400 mb-1">
                              {bowler.name}
                            </p>
                            <p className="text-sm text-gray-700 dark:text-gray-300">
                              {bowler.insight}
                            </p>
                          </div>
                        ))
                    ) : (
                      <p className="text-gray-500">No insights available.</p>
                    )}
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

// Partnerships Component
const Partnerships = ({ partnerships }: any) => {
  // Format partnerships data for visualization
  const partnershipData =
    partnerships?.map((p: any) => ({
      name: `${p.batsmen[0]} & ${p.batsmen[1]}`,
      runs: p.runs,
      balls: p.balls,
      run_rate: p.run_rate,
    })) || [];

  return (
    <motion.div
      className="space-y-6"
      variants={staggerContainer}
      initial="hidden"
      animate="visible"
    >
      <Card className="overflow-hidden border-none shadow-lg">
        <CardHeader className="pb-2 border-b border-gray-100 dark:border-gray-800">
          <CardTitle className="text-lg font-semibold text-blue-700 dark:text-blue-400">
            Partnership Analysis
          </CardTitle>
          <CardDescription>Insights on batting partnerships</CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          {/* Partnerships Chart */}
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-4">
              Partnership Contributions
            </h3>

            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={partnershipData}
                  margin={{ top: 10, right: 10, left: 10, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis
                    dataKey="name"
                    angle={-45}
                    textAnchor="end"
                    height={70}
                  />
                  <YAxis
                    label={{
                      value: "Runs",
                      angle: -90,
                      position: "insideLeft",
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      color: "#000000",
                      backgroundColor: "rgba(255, 255, 255, 0.95)",
                      borderRadius: "8px",
                      boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
                      border: "1px solid #e0e0e0",
                    }}
                    formatter={(value: any, name, props) => {
                      if (name === "strike_rate")
                        return [`${value.toFixed(2)}`, "Strike Rate"];
                      return [value, name];
                    }}
                  />
                  <Bar dataKey="runs" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Partnerships Table */}
          <div>
            <h3 className="text-lg font-medium mb-4">Partnership Details</h3>

            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Partnership</TableHead>
                  <TableHead className="text-right">Runs</TableHead>
                  <TableHead className="text-right">Balls</TableHead>
                  <TableHead className="text-right">Run Rate</TableHead>
                  {/* <TableHead className="text-right">Wicket</TableHead> */}
                </TableRow>
              </TableHeader>
              <TableBody>
                {partnerships && partnerships.length > 0 ? (
                  partnerships.map((partnership: any, index: any) => (
                    <TableRow key={index}>
                      <TableCell className="font-medium">
                        {partnership.batsmen[0]} & {partnership.batsmen[1]}
                      </TableCell>
                      <TableCell className="text-right">
                        {partnership.runs}
                      </TableCell>
                      <TableCell className="text-right">
                        {partnership.balls}
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge
                          className={getBadgeColorForStrikeRate(
                            partnership.run_rate || 0
                          )}
                        >
                          {(partnership.run_rate || 0).toFixed(1)}
                        </Badge>
                      </TableCell>
                      {/* <TableCell className="text-right">
                        {partnership.wicket ? (
                          <Badge variant="outline" className="bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-400">
                            {partnership.wicket}
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400">
                            Not Out
                          </Badge>
                        )}
                      </TableCell> */}
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell
                      colSpan={5}
                      className="text-center py-4 text-gray-500"
                    >
                      No partnership data available.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>

            {/* Partnership Insights */}
            <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/10 rounded-lg border border-blue-100 dark:border-blue-900">
              <div className="flex items-start">
                <TrendingUpIcon className="h-5 w-5 text-blue-500 mr-2 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-blue-700 dark:text-blue-400">
                    Key Partnership Insights
                  </p>
                  <div className="text-sm text-blue-700 dark:text-blue-300 mt-2 space-y-2">
                    {partnerships && partnerships.length > 0 ? (
                      <>
                        <p>
                          Highest Partnership:{" "}
                          {Math.max(...partnerships.map((p: any) => p.runs))}{" "}
                          runs
                          {partnerships.find(
                            (p: any) =>
                              p.runs ===
                              Math.max(...partnerships.map((p: any) => p.runs))
                          ) &&
                            ` (${
                              partnerships.find(
                                (p: any) =>
                                  p.runs ===
                                  Math.max(
                                    ...partnerships.map((p: any) => p.runs)
                                  )
                              ).batsmen[0]
                            } & 
                            ${
                              partnerships.find(
                                (p: any) =>
                                  p.runs ===
                                  Math.max(
                                    ...partnerships.map((p: any) => p.runs)
                                  )
                              ).batsmen[1]
                            })`}
                        </p>
                        <p>
                          Fastest Partnership:
                          {partnerships.find(
                            (p: any) =>
                              p.run_rate ===
                              Math.max(
                                ...partnerships.map((p: any) => p.run_rate)
                              )
                          ) &&
                            ` ${partnerships
                              .find(
                                (p: any) =>
                                  p.run_rate ===
                                  Math.max(
                                    ...partnerships.map((p: any) => p.run_rate)
                                  )
                              )
                              .run_rate.toFixed(1)} SR
                            (${
                              partnerships.find(
                                (p: any) =>
                                  p.run_rate ===
                                  Math.max(
                                    ...partnerships.map((p: any) => p.run_rate)
                                  )
                              ).batsmen[0]
                            } & 
                            ${
                              partnerships.find(
                                (p: any) =>
                                  p.run_rate ===
                                  Math.max(
                                    ...partnerships.map((p: any) => p.run_rate)
                                  )
                              ).batsmen[1]
                            })`}
                        </p>
                        <p>Total Partnerships: {partnerships.length}</p>
                      </>
                    ) : (
                      <p>No partnership data available for analysis.</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

const CricketAnalytics = ({ data }: any) => {
  const [activeTab, setActiveTab] = useState("summary");
  const [activeInnings, setActiveInnings] = useState(0);

  // Get the data for the active innings
  const inningsData = data?.innings_data?.[activeInnings] || {};

  // Get team names for tab labels
  const innings1Team = data?.innings_data?.[0]?.batting_team || "Team 1";
  const innings2Team = data?.innings_data?.[0]?.bowling_team || "Team 2";

  // Extract necessary data for the active innings
  const processedData = {
    batting_team: inningsData.batting_team || "Team",
    bowling_team: inningsData.bowling_team || "Opponent",
    batters_analysis: inningsData.batters_analysis || [],
    bowlers_analysis: inningsData.bowlers_analysis || [],
    partnerships: inningsData.partnerships || [],
    match_summary: inningsData.match_summary || {},
    batting_momentum: inningsData.batting_momentum || 50,
    bowling_momentum: inningsData.bowling_momentum || 50,
    recommendations: parseRecommendations(data.raw_recommendations) || [],
    momentum: inningsData.batting_momentum || 50,
    over_metrics: inningsData.over_metrics || [],
  };

  // Format run rate by over for charting
  const runRateByOver: any = {};
  if (processedData.over_metrics && processedData.over_metrics.length > 0) {
    processedData.over_metrics.forEach((metric: any) => {
      runRateByOver[metric.Over] = metric.Over_Run_Rate;
    });

    // Add to match summary
    processedData.match_summary.run_rate_by_over = runRateByOver;
  }

  return (
    <motion.div
      className="space-y-6 py-4 text-gray-800 dark:text-gray-200"
      variants={staggerContainer}
      initial="hidden"
      animate="visible"
    >
      {/* Innings tabs */}
      <Tabs
        defaultValue="innings1"
        className="w-full mb-6"
        onValueChange={(value) =>
          setActiveInnings(value === "innings1" ? 0 : 1)
        }
      >
        <TabsList className="grid grid-cols-2 bg-blue-50 dark:bg-blue-900/20 p-0 rounded-lg">
          <TabsTrigger
            value="innings1"
            className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2 font-semibold"
          >
            {innings1Team} (1st Innings)
          </TabsTrigger>
          <TabsTrigger
            value="innings2"
            className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2 font-semibold"
            disabled={!data?.innings_data?.[1]}
          >
            {innings2Team} (2nd Innings)
          </TabsTrigger>
        </TabsList>
      </Tabs>

      {/* Analysis tabs */}
      <motion.div
        key={`innings-${activeInnings}`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <Tabs
          defaultValue="summary"
          className="w-full"
          onValueChange={setActiveTab}
        >
          <TabsList className="grid grid-cols-5 mb-6 bg-gray-100 dark:bg-gray-800 p-0 rounded-lg">
            <TabsTrigger
              value="summary"
              className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2"
            >
              Summary
            </TabsTrigger>
            <TabsTrigger
              value="batting"
              className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2"
            >
              Batting
            </TabsTrigger>
            <TabsTrigger
              value="bowling"
              className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2"
            >
              Bowling
            </TabsTrigger>
            <TabsTrigger
              value="partnerships"
              className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2"
            >
              Partnerships
            </TabsTrigger>
            <TabsTrigger
              value="momentum"
              className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2"
            >
              Momentum
            </TabsTrigger>
          </TabsList>

          <motion.div
            key={`${activeInnings}-${activeTab}`}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <TabsContent value="summary" className="mt-0">
              <MatchSummary
                summary={processedData.match_summary}
                batting_momentum={processedData.batting_momentum}
                bowling_momentum={processedData.bowling_momentum}
                recommendations={processedData.recommendations}
              />
            </TabsContent>

            <TabsContent value="batting" className="mt-0">
              <BattersAnalysis batters={processedData.batters_analysis} />
            </TabsContent>

            <TabsContent value="bowling" className="mt-0">
              <BowlersAnalysis bowlers={processedData.bowlers_analysis} />
            </TabsContent>

            <TabsContent value="partnerships" className="mt-0">
              <Partnerships partnerships={processedData.partnerships} />
            </TabsContent>

            <TabsContent value="momentum" className="mt-0">
              <PredictionsTab
                summary={processedData.match_summary}
                batting_momentum={processedData.batting_momentum}
                bowling_momentum={processedData.bowling_momentum}
              />
            </TabsContent>
          </motion.div>
        </Tabs>
      </motion.div>
    </motion.div>
  );
};

export default CricketAnalytics;
