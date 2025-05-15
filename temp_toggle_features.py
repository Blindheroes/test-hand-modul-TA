 def toggle_features(self, use_threshold_adaptation=None, use_temporal_filtering=None):
      """
        Toggle the threshold adaptation and temporal filtering features on or off

        Args:
            use_threshold_adaptation: Boolean to enable/disable threshold adaptation, or None to leave unchanged
            use_temporal_filtering: Boolean to enable/disable temporal filtering, or None to leave unchanged
        """
       if use_threshold_adaptation is not None:
            self.use_threshold_adaptation = use_threshold_adaptation

        if use_temporal_filtering is not None:
            self.use_temporal_filtering = use_temporal_filtering

        # Clear gesture history when turning off temporal filtering
        if use_temporal_filtering is False:
            for gesture_name in self.gesture_history:
                self.gesture_history[gesture_name].clear()
