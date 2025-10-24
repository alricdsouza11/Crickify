import { useEffect, useState } from "react";
import { Match } from "../types/match";
import BattingCard from "./BattingCard";
import BowlingCard from "./BowlingCard";
import YetToBat from "./YetToBat";
import FallOfWickets from "./FallOfWickets";
import Image from "next/image";
import { motion } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  TrophyIcon,
  CalendarIcon,
  MapPinIcon,
  InfoIcon,
  CoinsIcon,
  AlertTriangleIcon,
  TrendingUpIcon,
  ZapIcon,
  TargetIcon,
  AlertCircleIcon,
  BarChartIcon,
  CircleIcon,
  PercentIcon,
  ActivityIcon,
  UserIcon,
} from "lucide-react";
import CricketAnalytics from "./CricketAnalytics";
import Recommendations from "./Recommendations";

interface ScoreCardProps {
  match: Match;
}

export default function ScoreCard({ match }: ScoreCardProps) {
  const [activeInnings, setActiveInnings] = useState(0);
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<any>(null);

  // Get innings data from the match object
  const innings1 = match.innings[0];
  const innings2 = match.innings.length > 1 ? match.innings[1] : null;

  const fadeInUp = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 },
    },
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

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch("http://127.0.0.1:5000/report");
        const responseData = await response.json();
        setData(responseData);
        setError(null);
      } catch (err: any) {
        console.error("Error fetching cricket data:", err);
        setError(err.response?.data || { message: err.message });
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <motion.div
        className="flex flex-col items-center justify-center py-16 space-y-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="relative w-16 h-16">
          <motion.div
            className="absolute top-0 left-0 w-full h-full border-4 border-blue-200 rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
          />
          <motion.div
            className="absolute top-0 left-0 w-full h-full border-t-4 border-blue-600 rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
          />
        </div>
        <p className="text-blue-600 font-medium">Loading match analysis...</p>
      </motion.div>
    );
  }

  if (error) {
    return (
      <motion.div
        className="rounded-lg border border-red-200 bg-red-50 dark:bg-red-900/10 dark:border-red-800 p-6 text-center"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <motion.div
          initial={{ scale: 0.8 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 400, damping: 10 }}
        >
          <AlertTriangleIcon className="mx-auto h-12 w-12 text-red-400 dark:text-red-300 mb-4" />
        </motion.div>
        <h3 className="text-lg font-semibold text-red-700 dark:text-red-300 mb-2">
          Error loading match data
        </h3>
        <p className="text-red-600 dark:text-red-200">
          {error.message || "Unknown error occurred"}
        </p>
        <p className="mt-4 text-sm text-red-500 dark:text-red-300">
          Please check your connection and try again later
        </p>
      </motion.div>
    );
  }

  if (!data) {
    return (
      <motion.div
        className="text-center py-12"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        No data available
      </motion.div>
    );
  }

  // Extract data from the API response for the current innings
  const currentInningsData =
    data.innings_data && data.innings_data.length > 0
      ? data.innings_data[activeInnings] || data.innings_data[0]
      : null;

  // Get the top batter from current innings data
  const topBatter =
    currentInningsData?.batters_analysis.sort(
      (a: any, b: any) => b.live_stats.runs - a.live_stats.runs
    )[0] || null;

  // Get the best bowler from current innings data - criteria: lowest economy with at least 1 wicket
  const bestBowler =
    currentInningsData?.bowlers_analysis.sort((a: any, b: any) => {
      if (a.live_stats.wickets === b.live_stats.wickets) {
        return a.live_stats.economy - b.live_stats.economy;
      }
      return b.live_stats.wickets - a.live_stats.wickets;
    })[0] || null;

  // Get current partnerships
  const currentPartnerships = currentInningsData?.partnerships || [];
  const latestPartnership =
    currentPartnerships.length > 0
      ? currentPartnerships[currentPartnerships.length - 1]
      : null;

  // Calculate momentum
  const momentum = currentInningsData?.batting_momentum || 50;

  // Handle probabilities from the data or set defaults if not available
  const probabilities = data.probabilities || {
    boundary: 0.25,
    dot_ball: 0.35,
    wicket: 0.05,
  };

  return (
    <motion.div
      className="w-full dark:text-white text-gray-800"
      initial="hidden"
      animate="visible"
      variants={fadeInUp}
    >
      <Tabs defaultValue="SUMMARY" className="w-full">
        <TabsList className="w-full grid grid-cols-2 md:grid-cols-4 mb-6">
          <TabsTrigger
            className="cursor-pointer data-[state=active]:bg-white dark:data-[state=active]:bg-gray-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2 px-4 w-fit mx-auto rounded-lg shadow-md"
            value="SUMMARY"
          >
            Summary
          </TabsTrigger>
          <TabsTrigger
            className="cursor-pointer data-[state=active]:bg-white dark:data-[state=active]:bg-gray-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2 px-4 w-fit mx-auto rounded-lg shadow-md"
            value="SCORECARD"
          >
            Scorecard
          </TabsTrigger>
          <TabsTrigger
            className="cursor-pointer data-[state=active]:bg-white dark:data-[state=active]:bg-gray-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2 px-4 w-fit mx-auto rounded-lg shadow-md"
            value="ANALYSIS"
          >
            Analysis
          </TabsTrigger>
          <TabsTrigger
            className="cursor-pointer data-[state=active]:bg-white dark:data-[state=active]:bg-gray-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2 px-4 w-fit mx-auto rounded-lg shadow-md"
            value="RECOMMENDATIONS"
          >
            Recommendations
          </TabsTrigger>
        </TabsList>

        <TabsContent value="SCORECARD" className="mt-2">
          <Card className="overflow-hidden border-none shadow-lg">
            <div className="bg-blue-50 dark:bg-blue-900/20 border-b border-blue-100 dark:border-blue-800">
              <Tabs
                value={activeInnings === 0 ? "innings1" : "innings2"}
                className="w-full"
                onValueChange={(value) =>
                  setActiveInnings(value === "innings1" ? 0 : 1)
                }
              >
                <TabsList className="w-full grid grid-cols-2 bg-transparent">
                  <TabsTrigger
                    value="innings1"
                    className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 rounded-none border-b-2 border-transparent data-[state=active]:border-blue-600 dark:data-[state=active]:border-blue-400"
                  >
                    {innings1.team.name} (1st Innings)
                  </TabsTrigger>
                  <TabsTrigger
                    value="innings2"
                    className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 rounded-none border-b-2 border-transparent data-[state=active]:border-blue-600 dark:data-[state=active]:border-blue-400"
                    disabled={!innings2 || data.innings_data.length < 2}
                  >
                    {innings2 && innings2.team.name} (2nd Innings)
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            <CardContent className="p-6">
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
                key={activeInnings}
              >
                {activeInnings === 0 ? (
                  <>
                    <BattingCard
                      battersAnalysis={
                        currentInningsData?.batters_analysis || []
                      }
                      partnerships={currentInningsData?.partnerships || []}
                      matchSummary={currentInningsData?.match_summary || {}}
                      battingTeam={
                        currentInningsData?.batting_team || innings1.team.name
                      }
                    />
                    <div className="mt-8">
                      <BowlingCard
                        bowlersAnalysis={
                          currentInningsData?.bowlers_analysis || []
                        }
                        bowlingTeam={
                          currentInningsData?.bowling_team || innings1.team.name
                        }
                      />
                    </div>
                  </>
                ) : (
                  <>
                    {data.innings_data.length > 1 ? (
                      <>
                        <BattingCard
                          battersAnalysis={
                            data.innings_data[1]?.batters_analysis || []
                          }
                          partnerships={
                            data.innings_data[1]?.partnerships || []
                          }
                          matchSummary={
                            data.innings_data[1]?.match_summary || {}
                          }
                          battingTeam={data.innings_data[1]?.batting_team}
                        />
                        <div className="mt-8">
                          <BowlingCard
                            bowlersAnalysis={
                              data.innings_data[1]?.bowlers_analysis || []
                            }
                            bowlingTeam={data.innings_data[1]?.bowling_team}
                          />
                        </div>
                      </>
                    ) : (
                      <div className="text-center py-12">
                        <p className="text-gray-500 dark:text-gray-400">
                          Innings 2 data not available yet
                        </p>
                      </div>
                    )}
                  </>
                )}
              </motion.div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="SUMMARY" className="space-y-6 mt-2">
          <Card className="border-none shadow-lg overflow-hidden">
            <div className="bg-blue-50 dark:bg-blue-900/20 border-b border-blue-100 dark:border-blue-800">
              <Tabs
                value={activeInnings === 0 ? "innings1" : "innings2"}
                className="w-full"
                onValueChange={(value) =>
                  setActiveInnings(value === "innings1" ? 0 : 1)
                }
              >
                <TabsList className="w-full grid grid-cols-2 bg-transparent">
                  <TabsTrigger
                    value="innings1"
                    className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 rounded-none border-b-2 border-transparent data-[state=active]:border-blue-600 dark:data-[state=active]:border-blue-400"
                  >
                    {innings1.team.name} (1st Innings)
                  </TabsTrigger>
                  <TabsTrigger
                    value="innings2"
                    className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 rounded-none border-b-2 border-transparent data-[state=active]:border-blue-600 dark:data-[state=active]:border-blue-400"
                    disabled={!innings2 || data.innings_data.length < 2}
                  >
                    {innings2 && innings2.team.name} (2nd Innings)
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            <CardContent className="p-6">
              <motion.div
                className="flex justify-between items-center mb-8"
                variants={fadeInUp}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
                key={activeInnings}
              >
                <motion.div
                  className="flex items-center"
                  whileHover={{ scale: 1.02 }}
                  transition={{ type: "spring", stiffness: 400, damping: 10 }}
                >
                  <div className="relative h-20 w-20 mr-6">
                    <Image
                      src={`/${activeInnings === 0 ? "mi" : "srh"}.png`}
                      alt={
                        currentInningsData?.batting_team ||
                        (activeInnings === 0
                          ? innings1.team.shortName
                          : innings2?.team.shortName)
                      }
                      layout="fill"
                      className="object-contain filter drop-shadow-md"
                    />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold">
                      {activeInnings === 0
                        ? innings1.team.shortName
                        : innings2?.team.shortName}
                    </h2>
                    <p className="text-xl">
                      {currentInningsData?.match_summary?.total_runs || 0}/
                      {currentInningsData?.match_summary?.total_wickets || 0}
                      <span className="text-gray-500 dark:text-gray-400 text-base ml-1">
                        (
                        {Math.floor(currentInningsData?.match_summary?.overs) ||
                          0}
                        )
                      </span>
                    </p>
                  </div>
                </motion.div>

                <div className="text-center">
                  <div className="relative w-12 h-12 mx-auto bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center text-blue-600 dark:text-blue-400 font-bold">
                    VS
                  </div>
                </div>

                <motion.div
                  className="flex items-center"
                  whileHover={{ scale: 1.02 }}
                  transition={{ type: "spring", stiffness: 400, damping: 10 }}
                >
                  <div>
                    <h2 className="text-2xl font-bold text-right">
                      {activeInnings === 0
                        ? innings2?.team.shortName
                        : innings1.team.shortName}
                    </h2>
                    <p className="text-xl text-right">
                      {activeInnings === 0
                        ? (data.innings_data[1]?.match_summary?.total_runs ||
                            0) +
                          "/" +
                          (data.innings_data[1]?.match_summary?.total_wickets ||
                            0)
                        : (data.innings_data[0]?.match_summary?.total_runs ||
                            0) +
                          "/" +
                          (data.innings_data[0]?.match_summary?.total_wickets ||
                            0)}
                      <span className="text-gray-500 dark:text-gray-400 text-base ml-1">
                        (
                        {activeInnings === 0
                          ? Math.floor(
                              data.innings_data[1]?.match_summary?.overs
                            ) || 0
                          : Math.floor(
                              data.innings_data[0]?.match_summary?.overs
                            ) || 0}
                        )
                      </span>
                    </p>
                  </div>
                  <div className="relative h-20 w-20 ml-6">
                    <Image
                      src={`/${activeInnings === 0 ? "srh" : "mi"}.png`}
                      alt={
                        activeInnings === 0
                          ? innings2?.team.shortName || "Unknown Team"
                          : innings1.team.shortName || "Unknown Team"
                      }
                      layout="fill"
                      className="object-contain filter drop-shadow-md"
                    />
                  </div>
                </motion.div>
              </motion.div>

              <motion.div
                className="bg-gradient-to-r from-blue-500 to-blue-600 p-4 rounded-lg mb-8 shadow-md"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.5 }}
                key={`runrate-${activeInnings}`}
              >
                <p className="text-center font-medium text-white text-lg">
                  {activeInnings === 0 ? "" : "Current "}Run Rate:{" "}
                  {currentInningsData?.match_summary?.current_run_rate.toFixed(
                    2
                  ) || "0.00"}
                  <span className="ml-2 px-2 py-1 bg-blue-700 rounded-md text-sm">
                    Batting Momentum:{" "}
                    {(currentInningsData?.batting_momentum || 50).toFixed(1)}
                  </span>
                  <span className="ml-2 px-2 py-1 bg-blue-700 rounded-md text-sm">
                    Bowling Momentum:{" "}
                    {(currentInningsData?.bowling_momentum || 50).toFixed(1)}
                  </span>
                </p>
              </motion.div>

              <motion.div
                className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8"
                variants={staggerContainer}
                initial="hidden"
                animate="visible"
                key={`cards-${activeInnings}`}
              >
                <motion.div variants={fadeInUp}>
                  <Card className="border border-blue-100 dark:border-blue-900/30 shadow-md hover:shadow-lg transition-shadow duration-300">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg font-semibold text-blue-700 dark:text-blue-300 flex items-center">
                        <TrophyIcon className="h-5 w-5 mr-2" />
                        Top Performers
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {topBatter && (
                        <motion.div
                          className="flex items-center p-3 bg-green-50 dark:bg-green-900/10 rounded-md border border-green-100 dark:border-green-900/20"
                          whileHover={{ x: 5 }}
                          transition={{
                            type: "spring",
                            stiffness: 300,
                            damping: 20,
                          }}
                        >
                          <div className="mr-4 bg-green-100 dark:bg-green-800/30 rounded-full p-2 text-green-600 dark:text-green-400">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-5 w-5"
                              viewBox="0 0 20 20"
                              fill="currentColor"
                            >
                              <path d="M10 3.5a1.5 1.5 0 013 0V4a1 1 0 011 1v3a1 1 0 01-1 1H9a1 1 0 01-1-1V5a1 1 0 011-1v-.5a1.5 1.5 0 011.5-1.5z" />
                              <path d="M3 14s-1 0-1-1 1-4 6-4 6 3 6 4-1 1-1 1H3zm5-6a3 3 0 100-6 3 3 0 000 6z" />
                            </svg>
                          </div>
                          <div className="flex-1">
                            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                              Top Scorer
                            </p>
                            <h4 className="font-bold text-green-700 dark:text-green-400">
                              {topBatter.name}
                            </h4>
                            <div className="flex items-center">
                              <Badge
                                variant="outline"
                                className="bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800 mr-2"
                              >
                                {topBatter.live_stats.runs} (
                                {topBatter.live_stats.balls_faced})
                              </Badge>
                              <p className="text-xs text-gray-600 dark:text-gray-400">
                                SR:{" "}
                                {topBatter.live_stats.strike_rate.toFixed(2)} ·{" "}
                                {topBatter.live_stats.boundaries} boundaries
                              </p>
                            </div>
                          </div>
                        </motion.div>
                      )}

                      {bestBowler && bestBowler.live_stats.wickets > 0 && (
                        <motion.div
                          className="flex items-center p-3 bg-blue-50 dark:bg-blue-900/10 rounded-md border border-blue-100 dark:border-blue-900/20"
                          whileHover={{ x: 5 }}
                          transition={{
                            type: "spring",
                            stiffness: 300,
                            damping: 20,
                          }}
                        >
                          <div className="mr-4 bg-blue-100 dark:bg-blue-800/30 rounded-full p-2 text-blue-600 dark:text-blue-400">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-5 w-5"
                              viewBox="0 0 20 20"
                              fill="currentColor"
                            >
                              <path
                                fillRule="evenodd"
                                d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                                clipRule="evenodd"
                              />
                            </svg>
                          </div>
                          <div className="flex-1">
                            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                              Best Bowler
                            </p>
                            <h4 className="font-bold text-blue-700 dark:text-blue-400">
                              {bestBowler.name}
                            </h4>
                            <div className="flex items-center">
                              <Badge
                                variant="outline"
                                className="bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800 mr-2"
                              >
                                {bestBowler.live_stats.wickets} wicket
                                {bestBowler.live_stats.wickets !== 1 ? "s" : ""}
                              </Badge>
                              <p className="text-xs text-gray-600 dark:text-gray-400">
                                {bestBowler.live_stats.runs_conceded} runs ·{" "}
                                {bestBowler.live_stats.overs.toFixed(1)} overs
                              </p>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>

                <motion.div variants={fadeInUp}>
                  <Card className="border border-blue-100 dark:border-blue-900/30 shadow-md hover:shadow-lg transition-shadow duration-300">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg font-semibold text-blue-700 dark:text-blue-300 flex items-center">
                        <InfoIcon className="h-5 w-5 mr-2" />
                        Live Analytics
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {/* {latestPartnership && (
                        <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                          <ZapIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                          <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                            Partnership
                          </div>
                          <div className="flex-1 text-sm">
                            {latestPartnership.runs} Runs off{" "}
                            {latestPartnership.balls} Balls (
                            {(
                              (latestPartnership.runs /
                                latestPartnership.balls) *
                              6
                            ).toFixed(2)}{" "}
                            RPO)
                          </div>
                        </div>
                      )} */}

                      <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                        <ZapIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                        <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                          Inning Details
                        </div>
                        <div className="flex-1 text-sm">
                          {currentInningsData.innings_number === 1 ? (
                            <span>
                              {currentInningsData.innings_number}st Innings (
                              {Math.floor(
                                currentInningsData.match_summary.overs
                              )}{" "}
                              overs{""})
                            </span>
                          ) : (
                            <span>
                              {currentInningsData.innings_number}nd Innings (
                              {Math.floor(
                                currentInningsData.match_summary.overs
                              )}{" "}
                              overs{""})
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                        <TrendingUpIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                        <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                          Next Ball
                        </div>
                        <div className="flex-1 text-sm">
                          Boundary: {(probabilities.boundary * 100).toFixed(0)}%
                          | Dot: {(probabilities.dot_ball * 100).toFixed(0)}% |
                          Wicket: {(probabilities.wicket * 100).toFixed(0)}%
                        </div>
                      </div>

                      {currentInningsData?.match_summary && (
                        <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                          <TargetIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                          <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                            Innings Pace
                          </div>
                          <div className="flex-1 text-sm">
                            Projected:{" "}
                            {Math.round(
                              currentInningsData.match_summary
                                .current_run_rate * 20
                            )}{" "}
                            (20 overs)
                          </div>
                        </div>
                      )}

                      <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                        <AlertCircleIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                        <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                          Key Insight
                        </div>
                        <div className="flex-1 text-sm">
                          {activeInnings === 0
                            ? "Batters performing above their historical strike rates"
                            : "Bowlers struggling with economy rates higher than historical averages"}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
                <motion.div variants={fadeInUp}>
                  <Card className="border border-blue-100 dark:border-blue-900/30 shadow-md hover:shadow-lg transition-shadow duration-300">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg font-semibold text-blue-700 dark:text-blue-300 flex items-center">
                        <BarChartIcon className="h-5 w-5 mr-2" />
                        Scoring Breakdown
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                        <CircleIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                        <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                          Total Boundaries
                        </div>
                        <div className="flex-1 text-sm">
                          {currentInningsData?.batters_analysis?.reduce(
                            (acc: any, batter: any) =>
                              acc + (batter.live_stats.boundaries || 0),
                            0
                          ) || 21}{" "}
                          boundaries
                        </div>
                      </div>

                      <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                        <ZapIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                        <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                          Best Partnership
                        </div>
                        <div className="flex-1 text-sm">
                          {currentInningsData?.partnerships?.reduce(
                            (best: any, current: any) =>
                              current.runs > best.runs ? current : best,
                            { runs: 0, balls: 0, batsmen: [] }
                          )?.runs || 80}{" "}
                          runs (
                          {currentInningsData?.partnerships?.reduce(
                            (best: any, current: any) =>
                              current.runs > best.runs ? current : best,
                            { runs: 0, balls: 0, batsmen: [] }
                          )?.balls || 34}{" "}
                          balls)
                        </div>
                      </div>

                      <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                        <TrendingUpIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                        <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                          Best Over
                        </div>
                        <div className="flex-1 text-sm">
                          {Math.max(
                            ...(currentInningsData?.over_metrics?.map(
                              (over: any) => over.Over_Runs
                            ) || [22])
                          )}{" "}
                          runs (over{" "}
                          {currentInningsData?.over_metrics?.reduce(
                            (maxIdx: any, over: any, idx: any, arr: any) =>
                              over.Over_Runs > arr[maxIdx].Over_Runs
                                ? idx
                                : maxIdx,
                            0
                          ) + 1 || 4}
                          )
                        </div>
                      </div>

                      <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                        <UserIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                        <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                          Best Batsman
                        </div>
                        <div className="flex-1 text-sm">
                          {currentInningsData?.batters_analysis?.reduce(
                            (maxBatter: any, current: any) =>
                              current.live_stats.runs >
                              maxBatter.live_stats.runs
                                ? current
                                : maxBatter,
                            { name: "N/A", live_stats: { runs: 0 } }
                          ).name || "Ishan Kishan"}{" "}
                          (
                          {currentInningsData?.batters_analysis?.reduce(
                            (maxBatter: any, current: any) =>
                              current.live_stats.runs >
                              maxBatter.live_stats.runs
                                ? current
                                : maxBatter,
                            { live_stats: { runs: 0 } }
                          ).live_stats.runs || 84}{" "}
                          runs)
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
                <motion.div variants={fadeInUp}>
                  <Card className="border border-blue-100 dark:border-blue-900/30 shadow-md hover:shadow-lg transition-shadow duration-300">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg font-semibold text-blue-700 dark:text-blue-300 flex items-center">
                        <InfoIcon className="h-5 w-5 mr-2" />
                        Match Info
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                        <CalendarIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                        <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                          Date
                        </div>
                        <div className="flex-1 text-sm">{match.date}</div>
                      </div>

                      <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                        <MapPinIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                        <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                          Venue
                        </div>
                        <div className="flex-1 text-sm">
                          {match.venue || "MA Chidambaram Stadium, Chennai"}
                        </div>
                      </div>

                      <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                        <InfoIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                        <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                          Match Type
                        </div>
                        <div className="flex-1 text-sm">
                          {match.matchType} {match.matchNumber}
                        </div>
                      </div>

                      <div className="flex items-center p-2 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors duration-200">
                        <CoinsIcon className="h-4 w-4 text-blue-500 dark:text-blue-400 mr-4" />
                        <div className="w-24 text-sm font-medium text-gray-500 dark:text-gray-400">
                          Toss
                        </div>
                        <div className="flex-1 text-sm">
                          {match?.toss && match.toss.winner}, chose to{" "}
                          {match?.toss && match.toss.decision}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              </motion.div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="ANALYSIS" className="mt-2">
          <Card className="border-none shadow-lg overflow-hidden">
            <CricketAnalytics data={data} />
          </Card>
        </TabsContent>

        <TabsContent value="RECOMMENDATIONS" className="mt-2">
          <Card className="border-none shadow-lg overflow-hidden">
            <Recommendations data={data} />
          </Card>
        </TabsContent>
      </Tabs>
    </motion.div>
  );
}
