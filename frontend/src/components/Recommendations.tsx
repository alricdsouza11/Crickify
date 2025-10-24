import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ArrowRightIcon, CheckCircleIcon } from "lucide-react";
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

const parseRecommendations = (rawRecommendations: any) => {
  console.log(rawRecommendations);
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

const Recommendations = ({ data }: any) => {
  const [activeInnings, setActiveInnings] = useState(0);
  const [animateKey, setAnimateKey] = useState(0); // New state to trigger animations

  // Get the data for the active innings
  const inningsData = data?.innings_data?.[activeInnings] || {};

  // Get team names for tab labels
  const innings1Team = data?.innings_data?.[0]?.batting_team || "Team 1";
  const innings2Team = data?.innings_data?.[0]?.bowling_team || "Team 2";

  // Extract necessary data for the active innings
  const processedRec = parseRecommendations(data.raw_recommendations) || [];

  // Handle tab change with animation reset
  const handleTabChange = (value: string) => {
    setActiveInnings(value === "innings1" ? 0 : 1);
    setAnimateKey((prev) => prev + 1); // Change key to force re-mount and trigger animation
  };

  return (
    <motion.div
      className="space-y-6 py-4 text-gray-800 dark:text-gray-200"
      variants={staggerContainer}
      initial="visible" // Changed from "hidden" to "visible" so initial load doesn't animate
      animate="visible"
    >
      {/* Innings tabs */}
      <Tabs
        defaultValue="innings1"
        className="w-full mb-6"
        onValueChange={handleTabChange} // Use the new handler
      >
        <TabsList className="grid grid-cols-2 bg-blue-50 dark:bg-blue-900/20 p-0 rounded-lg">
          <TabsTrigger
            value="innings1"
            className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2 font-semibold"
          >
            {innings1Team}
          </TabsTrigger>
          <TabsTrigger
            value="innings2"
            className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400 py-2 font-semibold"
            // disabled={!data?.innings_data?.[1]}
          >
            {innings2Team}
          </TabsTrigger>
        </TabsList>
      </Tabs>

      <AnimatePresence mode="wait">
        {data?.innings_data?.length === 2 ? (
          activeInnings === 1 ? (
            <motion.div
              key={`innings2-${animateKey}`} // Key changes to force remount
              className="mt-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Card className="border border-blue-100 dark:border-blue-900">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base font-medium text-blue-700 dark:text-blue-400 flex items-center">
                    <CheckCircleIcon className="h-4 w-4 mr-2" />
                    Batting Team Smart Recommendations (2nd Innings - Live)
                  </CardTitle>
                  <CardDescription>
                    Based on current match analytics
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <motion.div
                    className="space-y-3"
                    variants={staggerContainer}
                    initial="hidden"
                    animate="visible"
                  >
                    {processedRec &&
                      processedRec.slice(0, 5).map((rec: any, i: any) => (
                        <motion.div
                          key={i}
                          className="flex items-start p-3 rounded-lg bg-blue-50 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-800"
                          variants={fadeInUp} // Use fadeInUp variant
                          whileHover={{ x: 3 }}
                        >
                          <ArrowRightIcon className="h-5 w-5 text-blue-500 dark:text-blue-400 mt-0.5 mr-2 flex-shrink-0" />
                          <p className="text-sm text-gray-700 dark:text-gray-300">
                            {rec}
                          </p>
                        </motion.div>
                      ))}
                  </motion.div>
                </CardContent>
              </Card>
            </motion.div>
          ) : (
            <motion.div
              key={`innings1-${animateKey}`} // Key changes to force remount
              className="mt-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Card className="border border-blue-100 dark:border-blue-900">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base font-medium text-blue-700 dark:text-blue-400 flex items-center">
                    <CheckCircleIcon className="h-4 w-4 mr-2" />
                    Bowling Team Smart Recommendations (2nd Innings - Live)
                  </CardTitle>
                  <CardDescription>
                    Based on current match analytics
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <motion.div
                    className="space-y-3"
                    variants={staggerContainer}
                    initial="hidden"
                    animate="visible"
                  >
                    {processedRec &&
                      processedRec.slice(5, 10).map((rec: any, i: any) => (
                        <motion.div
                          key={i}
                          className="flex items-start p-3 rounded-lg bg-blue-50 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-800"
                          variants={fadeInUp} // Use fadeInUp variant
                          whileHover={{ x: 3 }}
                        >
                          <ArrowRightIcon className="h-5 w-5 text-blue-500 dark:text-blue-400 mt-0.5 mr-2 flex-shrink-0" />
                          <p className="text-sm text-gray-700 dark:text-gray-300">
                            {rec}
                          </p>
                        </motion.div>
                      ))}
                  </motion.div>
                </CardContent>
              </Card>
            </motion.div>
          )
        ) : activeInnings === 0 ? (
          <motion.div
            key={`single-innings1-${animateKey}`} // Key changes to force remount
            className="mt-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="border border-blue-100 dark:border-blue-900">
              <CardHeader className="pb-2">
                <CardTitle className="text-base font-medium text-blue-700 dark:text-blue-400 flex items-center">
                  <CheckCircleIcon className="h-4 w-4 mr-2" />
                  Batting Team Smart Recommendations (1st Innings - Live)
                </CardTitle>
                <CardDescription>
                  Based on current match analytics
                </CardDescription>
              </CardHeader>
              <CardContent>
                <motion.div
                  className="space-y-3"
                  variants={staggerContainer}
                  initial="hidden"
                  animate="visible"
                >
                  {processedRec &&
                    processedRec.slice(0, 5).map((rec: any, i: any) => (
                      <motion.div
                        key={i}
                        className="flex items-start p-3 rounded-lg bg-blue-50 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-800"
                        variants={fadeInUp} // Use fadeInUp variant
                        whileHover={{ x: 3 }}
                      >
                        <ArrowRightIcon className="h-5 w-5 text-blue-500 dark:text-blue-400 mt-0.5 mr-2 flex-shrink-0" />
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          {rec}
                        </p>
                      </motion.div>
                    ))}
                </motion.div>
              </CardContent>
            </Card>
          </motion.div>
        ) : (
          <motion.div
            key={`single-innings2-${animateKey}`} // Key changes to force remount
            className="mt-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="border border-blue-100 dark:border-blue-900">
              <CardHeader className="pb-2">
                <CardTitle className="text-base font-medium text-blue-700 dark:text-blue-400 flex items-center">
                  <CheckCircleIcon className="h-4 w-4 mr-2" />
                  Bowling Team Smart Recommendations (1st Innings - Live)
                </CardTitle>
                <CardDescription>
                  Based on current match analytics
                </CardDescription>
              </CardHeader>
              <CardContent>
                <motion.div
                  className="space-y-3"
                  variants={staggerContainer}
                  initial="hidden"
                  animate="visible"
                >
                  {processedRec &&
                    processedRec.slice(5, 10).map((rec: any, i: any) => (
                      <motion.div
                        key={i}
                        className="flex items-start p-3 rounded-lg bg-blue-50 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-800"
                        variants={fadeInUp} // Use fadeInUp variant
                        whileHover={{ x: 3 }}
                      >
                        <ArrowRightIcon className="h-5 w-5 text-blue-500 dark:text-blue-400 mt-0.5 mr-2 flex-shrink-0" />
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          {rec}
                        </p>
                      </motion.div>
                    ))}
                </motion.div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default Recommendations;
