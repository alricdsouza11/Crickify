import { Match } from "../types/match";
import Image from "next/image";
import { motion } from "framer-motion";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useEffect, useState } from "react";
import { AlertTriangleIcon } from "lucide-react";

interface MatchCardProps {
  match: Match;
  onClick: () => void;
}

export default function MatchCard({ match, onClick }: MatchCardProps) {
  const innings1 = match.innings[0];
  const innings2 = match.innings[1];

  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<any>(null);

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
        <p className="text-blue-600 font-medium">Loading match data...</p>
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

  return (
    <motion.div
      whileHover={{ y: -5 }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: "spring", stiffness: 400, damping: 17 }}
    >
      <Card
        className="overflow-hidden cursor-pointer border-none shadow-lg dark:text-white text-gray-800"
        onClick={onClick}
      >
        <CardHeader className="p-4 bg-gradient-to-r from-blue-600 to-blue-500 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-medium">IPL · {match.date}</h2>
              <Badge
                variant="outline"
                className="mt-1 text-xs text-blue-100 border-blue-300 bg-blue-600/30"
              >
                {match.matchType} {match.matchNumber}
              </Badge>
            </div>
            <motion.div
              whileHover={{ scale: 1.1, rotate: 10 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
            >
              <svg
                className="h-8 w-8 text-white opacity-70"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <circle cx="12" cy="12" r="4" fill="currentColor" />
              </svg>
            </motion.div>
          </div>
        </CardHeader>

        <CardContent className="p-5">
          <div className="flex items-center justify-between mb-6">
            <motion.div
              className="flex items-center space-x-4"
              whileHover={{ x: 5 }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
            >
              <div className="relative h-14 w-14 bg-blue-50 dark:bg-blue-900/20 rounded-full p-1 shadow-sm">
                <Image
                  src={`/mi.png`}
                  alt={innings1.team.shortName}
                  layout="fill"
                  className="object-contain p-1"
                />
              </div>
              <div>
                <h3 className="font-bold">{innings1.team.shortName}</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {data.innings_data[0].match_summary.total_runs}/{data.innings_data[0].match_summary.total_wickets}
                  <span className="text-xs text-gray-500 dark:text-gray-500 ml-1">
                    ({Math.floor(data.innings_data[0].match_summary.overs)})
                  </span>
                </p>
              </div>
            </motion.div>

            <div className="flex h-8 items-center">
              <div className="w-px h-8 bg-gray-200 dark:bg-gray-700"></div>
            </div>

            <motion.div
              className="flex items-center space-x-4"
              whileHover={{ x: -5 }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
            >
              <div>
                <h3 className="font-bold text-right">
                  {innings2.team.shortName}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 text-right">
                  {data.innings_data[1]?.match_summary.total_runs || 0}/{data.innings_data[1]?.match_summary.total_wickets || 0}
                  <span className="text-xs text-gray-500 dark:text-gray-500 ml-1">
                    ({Math.floor(data.innings_data[1]?.match_summary.overs) || 0})
                  </span>
                </p>
              </div>
              <div className="relative h-14 w-14 bg-blue-50 dark:bg-blue-900/20 rounded-full p-1 shadow-sm">
                <Image
                  src={`/srh.png`}
                  alt={innings2.team.shortName}
                  layout="fill"
                  className="object-contain p-1"
                />
              </div>
            </motion.div>
          </div>
        </CardContent>

        <CardFooter className="pt-2 pb-4 px-5 border-t border-gray-100 dark:border-gray-800">
          <div className="w-full">
            <p className="text-sm font-medium text-center text-blue-600 dark:text-blue-400">
              {/* {match.result} */}
              {match.toss &&
                match.toss.winner +
                  " won the toss and chose to " +
                  match.toss.decision}
            </p>
          </div>
        </CardFooter>
      </Card>
    </motion.div>
  );
}
